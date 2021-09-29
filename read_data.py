import pandas as pd
from sklearn.model_selection import train_test_split


def data_split(path):
    data = pd.read_csv(path)
    content = data['review']
    label = data['label']
    train_content, test_content, train_label, test_label = train_test_split(content, label, test_size=0.2,
                                                                            random_state=100)
    length = len(test_label)


    print(111111)


if __name__ == '__main__':
    path = 'data/simplifyweibo_4_moods.csv'
    data_split(path)
