import os
import torch
import pandas as pd

root_path = os.path.abspath('.')
model_path = os.path.join(root_path, 'model')
data_path = os.path.join(root_path, 'data')
train_data_path = os.path.join(data_path, 'train')
test_data_path = os.path.join(data_path, 'test')

name = ["star", "comment"]

list = []

def txt2list(file_path, select):
    file_path = os.path.join(file_path, select)
    files = os.listdir(file_path)
    print(file_path)
    for file in files:
        path = os.path.join(file_path, file)
        f = open(path, 'r', encoding='UTF-8')
        n = len(file)
        data = []
        data.append(file[:n-4].split('_')[1])
        str = f.read(10000)
        data.append(str)
        list.append(data)
    
def list2csv(target_path, list):
    dataframe = pd.DataFrame(columns=name, data=list)
    dataframe.to_csv(target_path)

if __name__ == "__main__":
    txt2list(train_data_path, 'pos')
    txt2list(train_data_path, 'neg')
    list2csv("train.csv", list)

    list = []
    txt2list(test_data_path, 'pos')
    txt2list(test_data_path, 'neg')
    list2csv("test.csv", list)