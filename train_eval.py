import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from read_data import get_time_dif
from transformers import AdamW


def train(config, model, train_iter, test_iter, eval_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    total_batch = 0
    dev_test_loss = float('inf')
    last_improve = 0
    flag = False
    model.train()
    for epoch in range(config.num_epochs):
        print(f'{epoch}/{config.num_epochs}')
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()


















