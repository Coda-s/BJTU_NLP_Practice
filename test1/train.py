import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchtext


accuracys = []

def eval(iterator, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in iterator:
        feature, target = batch.text, batch.label
        feature.t_()

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
    step = 1
    for epoch in range(1, args.epochs+1):
        print("step : %d" % step)
        step += 1
        for batch in train_iterator:
            model.train()
            feature, target = batch.text, batch.label
            feature.t_()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            accuracy = 100.0 * corrects/batch.batch_size
            accuracys.append(accuracy)

            if(args.batch_num == 0):
                break
            else:
                args.batch_num -= 1
        eval(validation_iterator, model, args)

    # 保存模型
    torch.save(model, "./model/model.pkl")

    
    n = len(accuracys)
    plt.switch_backend('agg')
    plt.ylim(0, 100)
    plt.plot(range(0, n), accuracys)

    plt.savefig("matplotlib.png")
    print("Finish")

            

            