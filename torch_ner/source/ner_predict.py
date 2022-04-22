import os
import pickle

import torch
from transformers import BertTokenizer

from torch_ner.source.utils import load_pkl

clue_map_dic = {"game": "game", "address": "address", "government": "government", "book": "book", "name": "name",
                "position": "position", "company": "company", "movie": "movie", "organization": "organization",
                "scene": "scene"}

def get_entities_result(query, model_path):
    """进一步封装识别结果"""
    map_dic = clue_map_dic
    sentence_list, predict_labels = predict(query, model_path)
    if len(predict_labels) == 0:
        print("句子: {0}\t实体识别结果为空".format(query))
        return []
    entities = []
    if len(sentence_list) == len(predict_labels):
        result = _bio_data_handler(sentence_list, predict_labels, map_dic)
        if len(result) != 0:
            end = 0
            prefix_len = 0

            for word, label in result:
                sen = query.lower()[end:]
                begin = sen.find(word) + prefix_len
                end = begin + len(word)
                prefix_len = end
                if begin != -1:
                    ent = dict(value=query[begin:end], type=label, begin=begin, end=end)
                    entities.append(ent)
    return entities


def predict(sentence, model_path):
    """模型预测"""
    max_seq_length = 128
    if len(sentence) > max_seq_length:
        return list(sentence), []

    tokenizer = BertTokenizer.from_pretrained(model_path)
    # 获取句子的input_ids、token_type_ids、attention_mask
    result = tokenizer.encode_plus(sentence)
    input_ids, token_type_ids, attention_mask = result["input_ids"], result["token_type_ids"], result["attention_mask"]
    sentence_list = tokenizer.tokenize(sentence)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        token_type_ids.append(0)
        attention_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(token_type_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)

    # 单词在词典中的编码、区分两个句子的编码、指定对哪些词进行self-Attention操作
    input_ids = input_ids.to("cpu").unsqueeze(0)
    token_type_ids = token_type_ids.to("cpu").unsqueeze(0)
    attention_mask = attention_mask.to("cpu").unsqueeze(0)

    # 加载模型
    model = torch.load(os.path.join(model_path, "ner_model.ckpt"), map_location="cpu")
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    # 模型预测，不需要反向传播
    with torch.no_grad():
        predict_val = model.predict(input_ids, token_type_ids, attention_mask)

    label2id = load_pkl(os.path.join(model_path, "label2id.pkl"))
    id2label = {value: key for key, value in label2id.items()}

    predict_labels = []
    for i, label in enumerate(predict_val[0]):
        if i != 0 and i != len(predict_val[0]) - 1:
            predict_labels.append(id2label[label])

    return sentence_list, predict_labels


def _bio_data_handler(sentence, predict_label, map_dic):
    """根据标签序列提取出实体"""
    entities = []
    # 获取初始位置实体标签
    pre_label = predict_label[0]
    # 实体词初始化
    word = ""
    for i in range(len(sentence)):
        # 记录问句当前位置词的实体标签
        current_label = predict_label[i]
        # 若当前位置的实体标签是以B开头的，说明当前位置是实体开始位置
        if current_label.startswith('B'):
            # 当前位置所属标签类别与前一位置所属标签类别不相同且实体词不为空，则说明开始记录新实体，前面的实体需要加到实体结果中
            if pre_label[2:] is not current_label[2:] and word != "":
                entities.append([word, map_dic[pre_label[2:]]])
                # 将当前实体词清空
                word = ""
            # 并将当前的词加入到实体词中
            word += sentence[i]
            # 记录当前位置标签为前一位置标签
            pre_label = current_label
        # 若当前位置的实体标签是以I开头的，说明当前位置是实体中间位置，将当前词加入到实体词中
        elif current_label.startswith('I') or current_label.startswith('M'):
            word += sentence[i]
            pre_label = current_label
        elif current_label.startswith('E'):
            word += sentence[i]
            pre_label = current_label
            if pre_label[2:] is current_label[2:]:
                entities.append([word, map_dic[current_label[2:]]])
                # 将当前实体词清空
                word = ""
        # 若当前位置的实体标签是以O开头的，说明当前位置不是实体，需要将实体词加入到实体结果中
        elif current_label.startswith('O'):
            # 当前位置所属标签类别与前一位置所属标签类别不相同且实体词不为空，则说明开始记录新实体，前面的实体需要加到实体结果中
            if pre_label[2:] is not current_label[2:] and word != "":
                entities.append([word, map_dic[pre_label[2:]]])
            # 记录当前位置标签为前一位置标签
            pre_label = current_label
            # 并将当前的词加入到实体词中
            word = ""
        elif current_label.startswith('S'):
            word += sentence[i]
            pre_label = current_label
    # 收尾工作，遍历问句完成后，若实体刚好处于最末位置，将剩余的实体词加入到实体结果中
    if word != "":
        entities.append([word, map_dic[pre_label[2:]]])
    return entities


if __name__ == '__main__':
    # 模型存放路径
    clue_path = os.path.join(os.path.abspath('..'), 'output\\clue_ner\\20220417160929')
    sent = "当天晚上，孙晓凯和王浩天等5人回到大连。"
    # 使用训练好的模型进行预测
    entities = get_entities_result(sent,clue_path)
    print(entities)
