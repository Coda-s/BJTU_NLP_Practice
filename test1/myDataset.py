import os
import random

import pandas as pd

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
        
        #print(self.datas)

        def clean_str(str):
            str = str.replace("<br />", "", 10000)
            return str.strip()

        text_field.tokenize = lambda x: clean_str(x).split()
        fields = [("text", text_field), ("label", label_field)]

        if examples is None:
            self.datas = []
            examples = []

            print("Loading")
            
            self.load_data(train_data_path, "pos")
            self.load_data(train_data_path, "neg")
            

            examples.extend([data.Example.fromlist(dt, fields) for dt in self.datas])
        
        super(DS, self).__init__(examples, fields, **kwargs)
   
    @classmethod
    def splits(cls, text_field, label_field, validation_ratio=.1, shuffle=True, **kwargs):
        examples = cls(text_field, label_field, **kwargs).examples

        if shuffle: random.shuffle(examples)
        validation_index = -1 * int(validation_ratio * len(examples))
        return (cls(text_field, label_field, examples=examples[:validation_index], **kwargs),
                cls(text_field, label_field, examples=examples[validation_index:], **kwargs))   

    def load_data(self, file_path, select):
        file_path = os.path.join(file_path, select)
        files = os.listdir(file_path)        #print(file_path)
        for file in files:
            path = os.path.join(file_path, file)
            with open(path, 'r', encoding="UTF-8") as f:
                n = len(file)
                line = []
                str = f.read()
                line.append(str)
                self.datas.append(line)
                line.append(file[:n-4].split('_')[1])
                f.close()
