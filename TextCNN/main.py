import os
import model
import train
import argparse
import torch
import myDataset
import warnings
import torchtext.data as data 
from torchtext.vocab import Vectors

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="CNN text classifier")
# dataset
parser.add_argument("--batch-size", type=int, default=20)
parser.add_argument("--shuffle", action="store_true", default=False)
# model
parser.add_argument("--kernel-sizes", type=str, default="3,4,5")
parser.add_argument("--dim", type=int, default=100)
parser.add_argument("--kernel-num", type=int, default=5)
# train
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-num", type=int, default=50)
# other
parser.add_argument("--train", action="store_true", default=False)

args = parser.parse_args()

def save_vocab(vocab, path):
    f = open(path, "w")
    for i in vocab.itos:
        f.write("{}\n".format(i))
    f.close()
    print("Save vocab succesfully!")

def make_iterotor(text_field, label_field, train):
    train_data, validation_data = myDataset.DS.splits(text_field, label_field)
    
    cache = ".vector_cache"
    if not os.path.exists(cache):
        os.mkdir(cache)
    vectors = Vectors(name='./data/glove.42B.300d.txt', cache=cache)
    
    text_field.build_vocab(train_data, validation_data, vectors=vectors)
    label_field.build_vocab(train_data, validation_data)

    print(len(train_data))
    print(len(validation_data))

    train_iter, validation_iter = data.Iterator.splits(
        (train_data, validation_data),
        batch_sizes = (args.batch_size, len(validation_data)/50))
    return train_iter, validation_iter

if __name__ == "__main__":
   
    text_field = data.Field(lower=True, fix_length=500)
    label_field = data.Field(sequential=False)

    train_iter, validation_iter = make_iterotor(text_field, label_field, args.train)

    args.num = len(text_field.vocab)
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.class_num = len(label_field.vocab)-1
    args.weight_matrix = text_field.vocab.vectors

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    if args.train is True:
        # 训练
        cnn = model.TextCNN(args)
        train.train(train_iter, validation_iter, cnn, args)
    else:
        # 测试
        print("Testing...")
        cnn = torch.load("./model/model_{}.pkl".format(1))
        train.predict(validation_iter, cnn, args)
        

        for i in range(1, 100):
            print("model_{}".format(i))
            train.eval(validation_iter, cnn, args)