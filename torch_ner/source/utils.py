import pickle


def load_pkl(fp):
    """加载pkl文件"""
    with open(fp, 'rb') as f:
        data = pickle.load(f)
        return data

def load_file(fp: str, sep: str = None, name_tuple=None):
    """
    读取文件；
    若sep为None，按行读取，返回文件内容列表，格式为:[xxx,xxx,xxx,...]
    若不为None，按行读取分隔，返回文件内容列表，格式为: [[xxx,xxx],[xxx,xxx],...]
    """
    with open(fp, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if sep:
            if name_tuple:
                return map(name_tuple._make, [line.strip().split(sep) for line in lines])
            else:
                return [line.strip().split(sep) for line in lines]
        else:
            return lines

def save_pkl(data, fp):
    """保存pkl文件，数据序列化"""
    with open(fp, 'wb') as f:
        pickle.dump(data, f)