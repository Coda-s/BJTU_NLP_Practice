import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        
        num = # 单词数量
        dim = # 词向量长度
        kernel_num = # 卷积核数量
        kernel_sizes = [] # 卷积核大小 
        input_num = 1 # 输入数量
        output_num = 10 # 输出数量

        # 词向量
        self.embedding = nn.Embedding(num, dim)
        # 卷积层
        self.convs = nn.ModuleList([nn.Conv2d(input_num, kernel_num, (size, dim)) for size in kernel_sizes])
        # self.dropout = nn.Dropout(dropout)
        # 线性层
        self.fc = nn.Linear(len(kernel_sizes)*kernel_num, output_num)

    def forward(self, x):
         x = self.embedding(x)
         x = x.unsqueeze(1)
         x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
         x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]
         x = torch.cat(x, 1)
         logit = self.fc(x)
         return logit