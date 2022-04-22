import os
import logging

import torch
from transformers import AdamW, get_linear_schedule_with_warmup, BertConfig, BertTokenizer
from tensorboardX import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from tqdm import tqdm, trange

from torch_ner.source.logger import logger as logger
from torch_ner.source.config import Config
import torch_ner.source.conlleval as evaluate
from torch_ner.source.models import BERT_BiLSTM_CRF
from torch_ner.source.ner_processor import NerProcessor


class NerMain(object):
    def __init__(self):
        # 初始化系统配置、数据预处理
        self.config = Config()
        self.processor = NerProcessor()

    def train(self):
        """模型训练"""
        # 清理output/xxx目录，若output/xxx目录存在，将会被删除, 然后初始化输出目录
        self.processor.clean_output(self.config)

        # SummaryWriter构造函数
        writer = SummaryWriter(logdir=os.path.join(self.config.output_path, "eval"), comment="ner")

        # 如果显存不足，可以通过gradient_accumulation_steps梯度累计来解决
        # 假设原来的batch_size = 10, 数据总量为1000，那么一共需要100train_steps，同时一共进行100次梯度更新。
        # 若是显存不够，我们需要减小batch＿size，我们设置gradient_accumulation_steps = 2，设置batch＿size = 5，
        # 我们需要运行两次，才能在内存中放入10条数据，梯度更新的次数不变为100次，那么我们的train＿steps = 200
        if self.config.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.config.gradient_accumulation_steps))

        # 配置可用设备，没有指定使用哪一块gpu，则全部使用
        use_gpu = torch.cuda.is_available() and self.config.use_gpu
        device = torch.device('cuda' if use_gpu else self.config.device)
        self.config.device = device
        n_gpu = torch.cuda.device_count()
        logger.info(f"available device: {device}，count_gpu: {n_gpu}")

        logger.info("====================== Start Data Pre-processing ======================")
        # 读取训练数据获取标签
        label_list = self.processor.get_labels(config=self.config)
        self.config.label_list = label_list
        num_labels = len(label_list)
        logger.info(f"loading labels successful! the size is {num_labels}, label is: {','.join(list(label_list))}")

        # 获取label2id、id2label的映射
        label2id, id2label = self.processor.get_label2id_id2label(self.config.output_path, label_list=label_list)
        logger.info("loading label2id and id2label dictionary successful!")

        if self.config.do_train:
            # 初始化tokenizer(标记生成器)、bert_config、BERT_BiLSTM_CRF
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=self.config.do_lower_case)
            bert_config = BertConfig.from_pretrained('bert-base-chinese', num_labels=num_labels)
            model = BERT_BiLSTM_CRF.from_pretrained('bert-base-chinese', config=bert_config, need_birnn=self.config.need_birnn, rnn_dim=self.config.rnn_dim)

            model.to(device)
            logger.info("loading tokenizer、bert_config and bert_bilstm_crf model successful!")

            if use_gpu and n_gpu > 1:
                model = torch.nn.DataParallel(model)

            logger.info("starting load train data and data_loader...")
            # 获取训练样本、样本特征、TensorDataset信息
            train_examples, train_features, train_data = self.processor.get_dataset(self.config, tokenizer, mode="train")

            # 训练数据载入
            train_data_loader = DataLoader(train_data, batch_size=self.config.train_batch_size, sampler=RandomSampler(train_data))
            logger.info("loading train data_set and data_loader successful!")

            eval_examples, eval_features, eval_data = [], [], None
            if self.config.do_eval:
                logger.info("starting load eval data...")
                eval_examples, eval_features, eval_data = self.processor.get_dataset(self.config, tokenizer, mode="eval")
                logger.info("loading eval data_set successful!")
            logger.info("====================== End Data Pre-processing ======================")

            # 初始化模型参数优化器
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)

            # 初始化学习率优化器
            t_total = len(train_data_loader) // self.config.gradient_accumulation_steps * self.config.num_train_epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=t_total)
            logger.info("loading AdamW optimizer、Warmup LinearSchedule and calculate optimizer parameter successful!")

            logger.info("====================== Running training ======================")
            logger.info(
                f"Num Examples:  {len(train_data)}, Num Batch Step: {len(train_data_loader)}, "
                f"Num Epochs: {self.config.num_train_epochs}, Num scheduler steps：{t_total}")

            # 启用 BatchNormalization 和 Dropout
            model.train()
            global_step, tr_loss, logging_loss, best_f1 = 0, 0.0, 0.0, 0.0
            for ep in trange(int(self.config.num_train_epochs), desc="Epoch"):
                logger.info(f"########[Epoch: {ep}/{int(self.config.num_train_epochs)}]########")
                model.train()
                for step, batch in enumerate(tqdm(train_data_loader, desc="DataLoader")):
                    logging.info(f"####[Step: {step}/{len(train_data_loader)}]####")

                    batch = tuple(t.to(device) for t in batch)
                    input_ids, token_type_ids, attention_mask, label_ids = batch
                    outputs = model(input_ids, label_ids, token_type_ids, attention_mask)
                    loss = outputs

                    if use_gpu and n_gpu > 1:
                        # mean() to average on multi-gpu.
                        loss = loss.mean()

                    if self.config.gradient_accumulation_steps > 1:
                        loss = loss / self.config.gradient_accumulation_steps

                    # 反向传播
                    loss.backward()
                    tr_loss += loss.item()

                    # 优化器_模型参数的总更新次数，和上面的t_total对应
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        # 更新参数
                        optimizer.step()
                        scheduler.step()
                        # 梯度清零
                        model.zero_grad()
                        global_step += 1

                        if self.config.logging_steps > 0 and global_step % self.config.logging_steps == 0:
                            tr_loss_avg = (tr_loss - logging_loss) / self.config.logging_steps
                            writer.add_scalar("Train/loss", tr_loss_avg, global_step)
                            logging_loss = tr_loss
                # 模型验证

                if self.config.do_eval:
                    logger.info("====================== Running Eval ======================")
                    all_ori_tokens_eval = [f.ori_tokens for f in eval_features]
                    overall, by_type = self.evaluate(self.config, eval_data, model, id2label, all_ori_tokens_eval)

                    # add eval result to tensorboard
                    f1_score = overall.fscore
                    writer.add_scalar("Eval/precision", overall.prec, ep)
                    writer.add_scalar("Eval/recall", overall.rec, ep)
                    writer.add_scalar("Eval/f1_score", overall.fscore, ep)

                    # save the best performs model
                    if f1_score > best_f1:
                        logging.info(f"******** the best f1 is {f1_score}, save model !!! ********")
                        best_f1 = f1_score
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, 'module') else model
                        model_to_save.save_pretrained(self.config.output_path)
                        tokenizer.save_pretrained(self.config.output_path)

                        # Good practice: save your training arguments together with the trained model
                        torch.save(self.config, os.path.join(self.config.output_path, 'training_config.bin'))
                        torch.save(model, os.path.join(self.config.output_path, 'ner_model.ckpt'))
                        logging.info("training_args.bin and ner_model.ckpt save successful!")
            writer.close()
            logging.info("NER model training successful!!!")
        if self.config.do_test:
            tokenizer = BertTokenizer.from_pretrained(self.config.output_path, do_lower_case=self.config.do_lower_case)
            config = torch.load(os.path.join(self.config.output_path, 'training_config.bin'))
            model = BERT_BiLSTM_CRF.from_pretrained(self.config.output_path, need_birnn=self.config.need_birnn, rnn_dim=self.config.rnn_dim)
            model.to(device)

            test_examples, test_features, test_data = self.processor.get_dataset(config, tokenizer, mode="test")

            logging.info("====================== Running test ======================")
            logging.info(f"Num Examples:  {len(test_examples)}, Batch size: {config.eval_batch_size}")

            all_ori_tokens = [f.ori_tokens for f in test_features]
            all_ori_labels = [e.label.split(" ") for e in test_examples]
            test_sampler = SequentialSampler(test_data)
            test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=config.eval_batch_size)
            model.eval()

            pred_labels = []

            for b_i, (input_ids, token_type_ids, attention_mask, label_ids) in enumerate(
                    tqdm(test_data_loader, desc="TestDataLoader")):

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)

                with torch.no_grad():
                    logits = model.predict(input_ids, token_type_ids, attention_mask)

                for l in logits:
                    pred_label = []
                    for idx in l:
                        pred_label.append(id2label[idx])
                    pred_labels.append(pred_label)

            assert len(pred_labels) == len(all_ori_tokens) == len(all_ori_labels)

            with open(os.path.join(config.output_path, "token_labels_test.txt"), "w", encoding="utf-8") as f:
                for ori_tokens, ori_labels, prel in zip(all_ori_tokens, all_ori_labels, pred_labels):
                    for ot, ol, pl in zip(ori_tokens, ori_labels, prel):
                        if ot in ["[CLS]", "[SEP]"]:
                            continue
                        else:
                            f.write(f"{ot} {ol} {pl}\n")
                    f.write("\n")

    @staticmethod
    def evaluate(config: Config, data, model, id2label, all_ori_tokens):
        ori_labels, pred_labels = [], []
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        model.eval()
        sampler = SequentialSampler(data)
        data_loader = DataLoader(data, sampler=sampler, batch_size=config.train_batch_size)
        for b_i, (input_ids, token_type_ids, attention_mask, label_ids) in enumerate(tqdm(data_loader, desc="Evaluating")):
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            token_type_ids = token_type_ids.to(config.device)
            label_ids = label_ids.to(config.device)
            with torch.no_grad():
                logits = model.predict(input_ids, token_type_ids, attention_mask)

            for l in logits:
                pred_labels.append([id2label[idx] for idx in l])

            for l in label_ids:
                ori_labels.append([id2label[idx.item()] for idx in l])

        eval_list = []
        for ori_tokens, oril, prel in zip(all_ori_tokens, ori_labels, pred_labels):
            for ot, ol, pl in zip(ori_tokens, oril, prel):
                if ot in ["[CLS]", "[SEP]"]:
                    continue
                eval_list.append(f"{ot} {ol} {pl}\n")
            eval_list.append("\n")

        # eval the model
        counts = evaluate.evaluate(eval_list)
        evaluate.report(counts)

        # namedtuple('Metrics', 'tp fp fn prec rec fscore')
        overall, by_type = evaluate.metrics(counts)
        return overall, by_type

if __name__ == '__main__':
    NerMain().train()

