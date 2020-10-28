import os

path = os.getcwd()
model_path = path + '\\model'
data_path = path + '\\data'
train_data_path = data_path + '\\train'
test_data_path = data_path + '\\test'

print(os.listdir(train_data_path + '\\neg'))