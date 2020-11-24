import os
import argparse
import warnings

# pytorch
import torch

# torchtext
import torchtext.data as data

# my files
import mydataset
import model
import train

warnings.filterwarnings("ignore")

# global variables
train_path = "./data/train"
valid_path = "./data/valid"

# parameters
parser = argparse.ArgumentParser(description="BiLSTM_CRF")
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--embedding-dim", type=int, default=300)
parser.add_argument("--hidden-dim", type=int, default=200)
parser.add_argument("--learning-rate", type=float, default=0.03)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=100)


if __name__ == "__main__":
    args = parser.parse_args()

    args.START_TAG = "<START>"
    args.STOP_TAG = "<STOP>"

    text_field = data.Field(lower=True)
    label_field = data.Field(unk_token=None, pad_token=None)

    train_data = mydataset.DS(text_field, label_field, train_path)
    valid_data = mydataset.DS(text_field, label_field, valid_path)
    
    text_field.build_vocab(train_data, valid_data)
    label_field.build_vocab(train_data, valid_data, specials=[args.START_TAG, args.STOP_TAG])
    
    args.vocab_size = len(text_field.vocab)
    args.tagset_size = len(label_field.vocab)
    args.tag_to_idx = label_field.vocab.stoi
    args.word_to_idx = text_field.vocab.stoi
    

    # print("\nParameters:")
    # for attr, value in sorted(args.__dict__.items()):
    #     print("\t{}={}".format(attr.upper(), value))
    
    train_iter = data.Iterator(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    valid_iter = data.Iterator(dataset=valid_data, batch_size=args.batch_size, shuffle=True)

    model = model.BiLSTM_CRF(args)

    train.train(train_iter, valid_iter, model, args)