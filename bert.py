import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Config(object):
    """配置参数"""

    def __init__(self, path):
        self.dataset_path = path + '/simplifyweibo_4_moods.csv'
        self.class_list = [x.strip() for x in open(path + '/class.txt').readlines()]
        self.save_path = path + '/bert.pt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 2000
        self.num_classes = len(self.class_list)
        self.num_epochs = 100
        self.batch_size = 128
        self.pad_size = 103  # 添加['cls']
        self.learning_rate = 5e-5
        self.bert_path = './bert_model'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        _, pooled = self.bert(context, attention_mask=mask, return_dict=False)
        output = self.fc(pooled)
        return output






