import logging
import os

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from torch_ner.source.config import Config
from torch_ner.source.utils import load_pkl, save_pkl, load_file

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, token_type_ids, attention_mask, label_id, ori_tokens):
        """
        :param input_ids:       单词在词典中的编码
        :param attention_mask:  指定 对哪些词 进行self-Attention操作
        :param token_type_ids:  区分两个句子的编码（上句全为0，下句全为1）
        :param label_id:        标签的id
        """
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label_id = label_id
        self.ori_tokens = ori_tokens

class NerProcessor(object):

    @staticmethod
    def get_labels(config: Config):
        """读取训练数据获取标签"""
        label_pkl_path = os.path.join(config.output_path, "label_list.pkl")
        if os.path.exists(label_pkl_path):
            logging.info(f"loading labels info from {config.output_path}")
            labels = load_pkl(label_pkl_path)
        else:
            logging.info(f"loading labels info from train file and dump in {config.output_path}")
            tokens_list = load_file(config.train_file, sep=config.sep)
            labels = set([tokens[1] for tokens in tokens_list if len(tokens) == 2])

        if len(labels) == 0:
            ValueError("loading labels error, labels type not found in data file: {}".format(config.output_path))
        else:
            save_pkl(labels, label_pkl_path)

        return labels

    @staticmethod
    def get_label2id_id2label(output_path, label_list):
        """获取label2id、id2label的映射"""
        label2id_path = os.path.join(output_path, "label2id.pkl")
        if os.path.exists(label2id_path):
            label2id = load_pkl(label2id_path)
        else:
            label2id = {l: i for i, l in enumerate(label_list)}
            save_pkl(label2id, label2id_path)

        id2label = {value: key for key, value in label2id.items()}
        return label2id, id2label

    @staticmethod
    def clean_output(config: Config):
        """清理output_xxx目录，若output_xxx目录存在，将会被删除, 然后初始化输出目录"""
        if config.clean and config.do_train:
            logging.info(f"clear output dir: {config.output_path}")
            if os.path.exists(config.output_path):
                def del_file(path):
                    ls = os.listdir(path)
                    for i in ls:
                        c_path = os.path.join(path, i)
                        if os.path.isdir(c_path):
                            del_file(c_path)
                            os.rmdir(c_path)
                        else:
                            os.remove(c_path)

                try:
                    del_file(config.output_path)
                except Exception as e:
                    logging.error(e)
                    logging.error('pleace remove the files of output dir and data.conf')
                    exit(-1)

        # 初始化output目录
        if os.path.exists(config.output_path) and os.listdir(config.output_path) and config.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(config.output_path))

        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        if not os.path.exists(os.path.join(config.output_path, "eval")):
            os.makedirs(os.path.join(config.output_path, "eval"))

    def get_dataset(self, config: Config, tokenizer, mode="train"):
        """
        对指定数据集进行预处理，进一步封装数据，包括:
        examples：[InputExample(guid=index, text=text, label=label)]
        features：[InputFeatures( input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask,
                                  label_id=label_ids,
                                  ori_tokens=ori_tokens)]
        data： 处理完成的数据集, TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_label_ids)
        """
        if mode == "train":
            file_path = config.train_file
        elif mode == "eval":
            file_path = config.eval_file
        elif mode == "test":
            file_path = config.test_file
        else:
            raise ValueError("mode must be one of train, eval, or test")

        # 读取输入数据，进一步封装
        examples = self.get_input_examples(file_path, separator=config.sep)

        # 对输入数据进行特征转换
        features = self.convert_examples_to_features(config, examples, tokenizer)

        # 获取全部数据的特征，封装成TensorDataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_label_ids)

        return examples, features, data

    def get_input_examples(self, input_file, separator=" "):
        """通过读取输入数据，封装输入样本"""
        examples = []
        lines = self.read_data(input_file, separator=separator)
        for i, line in enumerate(lines):
            guid = str(i)
            text = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    @staticmethod
    def read_data(input_file, separator="\t"):
        """读取输入数据"""
        with open(input_file, "r", encoding="utf-8") as f:
            lines, words, labels = [], [], []
            for line in f.readlines():
                contends = line.strip()
                tokens = line.strip().split(separator)
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
            return lines

    @staticmethod
    def convert_examples_to_features(config: Config, examples, tokenizer):
        """对输入数据进行特征转换"""
        label_map = {label: i for i, label in enumerate(config.label_list)}
        max_seq_length = config.max_seq_length
        features = []
        for ex_index, example in enumerate(tqdm(examples, desc="convert examples")):
            example_text_list = example.text.split(" ")
            example_label_list = example.label.split(" ")

            assert len(example_text_list) == len(example_label_list)

            tokens, labels, ori_tokens = [], [], []
            word_piece = False
            for i, word in enumerate(example_text_list):
                # 防止wordPiece情况出现
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label = example_label_list[i]
                ori_tokens.append(word)
                # 单个字符不会出现wordPiece
                if len(token) == 1:
                    labels.append(label)
                else:
                    word_piece = True

            if word_piece:
                logging.info("Error tokens!!! skip this lines, the content is: %s" % " ".join(example_text_list))
                continue

            # 当句子长度大于自定义的最大句子长度时，删除多余的字符
            if len(tokens) >= max_seq_length - 1:
                # -2的原因是因为序列需要加一个句首和句尾标志
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                ori_tokens = ori_tokens[0:(max_seq_length - 2)]

            # 给序列加上句首和句尾标志, 统一将序列padding到max_length长度
            sen_code = tokenizer.encode_plus(tokens, add_special_tokens=True, max_length=max_seq_length, padding="max_length")
            ori_tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]

            input_ids, token_type_ids, attention_mask = sen_code["input_ids"], sen_code["token_type_ids"], sen_code[
                "attention_mask"]

            label_ids = [label_map["O"]] + [label_map[labels[i]] for i, token in enumerate(tokens)] + [label_map["O"]]
            label_ids.extend([label_map["O"]] * (max_seq_length - len(label_ids)))

            assert len(input_ids) == max_seq_length
            assert len(attention_mask) == max_seq_length
            assert len(token_type_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            if ex_index < 3:
                logging.info("****** Example ******")
                logging.info("guid: %s" % example.guid)
                logging.info("tokens: %s" % tokenizer.convert_ids_to_tokens(sen_code["input_ids"]))
                logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logging.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logging.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

            features.append(InputFeatures(input_ids=input_ids,
                                          token_type_ids=token_type_ids,
                                          attention_mask=attention_mask,
                                          label_id=label_ids,
                                          ori_tokens=ori_tokens))
        return features