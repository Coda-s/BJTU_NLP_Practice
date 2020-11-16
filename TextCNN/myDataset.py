import re
import os
import random

import pandas as pd
import nltk

import torch
from torchtext import data

root_path = os.path.abspath('.')
model_path = os.path.join(root_path, 'model')
data_path = os.path.join(root_path, 'data')
train_data_path = os.path.join(data_path, 'train')
test_data_path = os.path.join(data_path, 'test')

features = []
targets = []

class DS(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, examples=None, **kwargs):
        def clean_str(str):
            str = str.replace("<br />", "", 10000)
            str = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", str)
            str = re.sub(r"\'s", " \'s", str)
            str = re.sub(r"\'ve", " \'ve", str)
            str = re.sub(r"n\'t", " n\'t", str)
            str = re.sub(r"\'re", " \'re", str)
            str = re.sub(r"\'d", " \'d", str)
            str = re.sub(r"\'ll", " \'ll", str)
            str = re.sub(r",", " , ", str)
            str = re.sub(r"!", " ! ", str)
            str = re.sub(r"\(", " ( ", str)
            str = re.sub(r"\)", " ) ", str)
            str = re.sub(r"\?", " ? ", str)
            str = re.sub(r"\s{2,}", " ", str)

            return str.strip()
            #return nltk.word_tokenize(str)

        text_field.tokenize = lambda x: clean_str(x).split()
        fields = [("text", text_field), ("label", label_field)]

        if examples is None:
            examples = []

            
            print("Loading train data")
            # train_data
            self.datas = []
            self.load_data(train_data_path, "pos")
            self.load_data(train_data_path, "neg")
            examples.extend([data.Example.fromlist(dt, fields) for dt in self.datas])

            print("Loading validation data")
            # validation_data
            self.datas = []
            self.load_data(test_data_path, "pos")
            self.load_data(test_data_path, "neg")
            examples.extend([data.Example.fromlist(dt, fields) for dt in self.datas])


        super(DS, self).__init__(examples, fields)
   
    @classmethod
    def splits(cls, text_field, label_field, validation_ratio=.1, shuffle=True, **kwargs):
        examples = cls(text_field, label_field, **kwargs).examples

        if shuffle:
            random.shuffle(examples[:25000])
            random.shuffle(examples[25000:])
        
        return (cls(text_field, label_field, examples=examples[:25000], **kwargs),
                cls(text_field, label_field, examples=examples[25000:], **kwargs))

    def load_data(self, file_path, select):
        file_path = os.path.join(file_path, select)
        print(file_path)
        files = os.listdir(file_path)
        for file in files:
            path = os.path.join(file_path, file)
            with open(path, 'r', encoding="UTF-8") as f:
                n = len(file)
                line = []
                str = f.read()
                line.append(str)
                label = int(file[:n-4].split('_')[1])
                if(label > 5):
                    label = 'pos'
                else:
                    label = 'neg'
                line.append(label)
                self.datas.append(line)
                f.close()
