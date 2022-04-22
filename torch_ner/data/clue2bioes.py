import json
import glob
import os


def json2bioes(input_path):
    with open(input_path, 'r', encoding='utf-8') as f, open(
            './result/' + 'cluener.' + input_path.rstrip('json') + 'bioes', 'w', encoding='utf-8') as fout:
        sentence_num, char_num, entity_num = 0, 0, 0
        for line in f:
            sentence_num += 1

            line = json.loads(line.strip())
            text = line['text']
            label_entities = line.get('label', None)

            words = list(text)
            char_num += len(words)
            labels = ['O'] * len(words)

            if label_entities is not None:
                entity_num += len(label_entities)
                for key, value in label_entities.items():
                    for sub_name, sub_index in value.items():
                        for start_index, end_index in sub_index:
                            assert ''.join(words[start_index:end_index + 1]) == sub_name
                            if start_index == end_index:
                                labels[start_index] = 'S-' + key
                            else:
                                labels[start_index] = 'B-' + key
                                labels[start_index + 1:end_index] = ['I-' + key] * (len(sub_name) - 2)
                                labels[end_index] = 'E-' + key

            for idx in range(len(words)):
                fout.write(words[idx] + ' ' + labels[idx] + '\n')
            fout.write('\n')

        with open('result/info.txt', 'a', encoding='utf-8') as info:
            info.write(input_path + ':\n' + \
                       'sentence_num: ' + str(sentence_num) + \
                       '\tchar_num: ' + str(char_num) + \
                       '\tentity_num: ' + str(entity_num) + '\n\n')


def main():
    json_files = glob.glob('*.json')
    if not os.path.exists('result'):
        os.makedirs('result')
    for file in json_files:
        json2bioes(file)


if __name__ == "__main__":
    main()