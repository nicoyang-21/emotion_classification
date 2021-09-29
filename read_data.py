import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import Config as config

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def data_split(path):
    data = pd.read_csv(path)
    content = data['review']
    label = data['label']
    train_content, test_content, train_label, test_label = train_test_split(content, label, test_size=0.2,
                                                                            random_state=100)
    return train_content, test_content, train_label, test_label


def build_dataset(config):
    def load_dataset(content, label, pad_size=420):
        contents = []
        for line, label in tqdm(zip(content, label)):
            lin = line.strip()
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_id = config.tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_id) + [0] * (pad_size - len(token))
                    token_id += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_id = token_id[:pad_size]
                    seq_len = pad_size
            contents.append((token_id, label, seq_len, mask))
        return contents

    train_content, test_content, train_label, test_label = data_split(config.dataset_path)
    train = load_dataset(train_content, train_label)
    test = load_dataset(test_content, test_label)
    return train, test






if __name__ == '__main__':
    path = 'data/simplifyweibo_4_moods.csv'
    data_split(path)
