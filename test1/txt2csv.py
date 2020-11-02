import os
import pandas as pd

from torchtext import data

root_path = os.path.abspath('.')
model_path = os.path.join(root_path, 'model')
data_path = os.path.join(root_path, 'data')
train_data_path = os.path.join(data_path, 'train')
test_data_path = os.path.join(data_path, 'test')

name = ["star", "comment"]
list = []

class Mydataset(data.Dataset):
    def __init__(self, text_field, label_field, **kargs):
        txt2list(train_data_path, 'pos')
        txt2list(train_data_path, 'neg')
        self.data = list
















def change(str):
    str = str.replace("<br />", "", 10000)
    return str

# 将txt转为list
def txt2list(file_path, select):
    file_path = os.path.join(file_path, select)
    files = os.listdir(file_path)
    print(file_path)
    i = 10
    for file in files:
        path = os.path.join(file_path, file)
        with open(path, 'r', encoding="UTF-8") as f:
            n = len(file)
            data = []
            data.append(file[:n-4].split('_')[1])
            str = f.read()
            str = change(str)
            data.append(str)
            list.append(data)
            f.close()

# 将list转为csv
def list2csv(target_path, list):
    dataframe = pd.DataFrame(columns=name, data=list)
    dataframe.to_csv(target_path, encoding="UTF-8")

if __name__ == "__main__":
    #txt2list(train_data_path, 'pos')
    #txt2list(train_data_path, 'neg')
    #list2csv("./data/train.csv", list)

    #list = []
    #txt2list(test_data_path, 'pos')
    #txt2list(test_data_path, 'neg')
    #list2csv("test.csv", list)