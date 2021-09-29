import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel, BertTokenizer


class Config(object):
    """配置参数"""
    def __init__(self, path):
        self.dataset_path = path + '/simplifyweibo_4_moods.csv'
        self.class_list = [x.strip() for x in open(path + '/class.txt').readlines()]
        self.save_path = path + '/bert.pt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 3
        self.batch_size = 128
        self.pad_len = 421
        self.learning_rate = 5e-5
        self.bert_path = './bert_model'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768



