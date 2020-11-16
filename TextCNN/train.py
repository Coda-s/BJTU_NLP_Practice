import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchtext

def predict(iterator, model, args):
    model.eval()
    list = []
    for batch in iterator:
        feature, target = batch.text, batch.label
        feature.t_()
        target.sub_(1)

        logit = model(feature)
        #list.extend(logit[1].view(target.size()).data)
        for x in logit:
            if(x[0] < x[1]):
                list.append("pos")
            else:
                list.append("neg")
    print(list)
    

def eval(iterator, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in iterator:
        feature, target = batch.text, batch.label
        feature.t_()
        target.sub_(1)

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.item()

        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
    
    size = len(iterator.dataset)
    avg_loss /= size
    accuracy = 100.0*corrects/size
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy

def train(train_iterator, validation_iterator, model, args):
    print("Start training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(1, args.epochs+1):
        print("epoch : {}".format(epoch))
        for batch in train_iterator:
            model.train()
            feature, target = batch.text, batch.label
            feature.t_()
            target.sub_(1)

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
        print("validation result :")
        eval(validation_iterator, model, args)
        # 保存模型
        torch.save(model, "./model/model_{}.pkl".format(epoch))
    print("Finish")

            

            