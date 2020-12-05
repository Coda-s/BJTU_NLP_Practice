import os
import argparse
import warnings
import torchtext.data as data
from torchtext.vocab import Vectors

# my files
import mydataset
import model
import train

# parameters
parser = argparse.ArgumentParser(description="ESIM")
parser.add_argument("--batch-size", type=int, default=20)
parser.add_argument("--embedding-dim", type=int, default=300)
parser.add_argument("--hidden-dim", type=int, default=200)
parser.add_argument("--learning-rate", type=float, default=0.03)
parser.add_argument("--epoches", type=int, default=100)

args = parser.parse_args()

TRAIN_DATA_PATH = "./data/snli_1.0_train.jsonl"
VALID_DATA_PATH = "./data/snli_1.0_dev.jsonl"

def load_data(text_field, label_field):
    print("Loading data...")

    train_data = mydataset.DS(text_field, label_field, TRAIN_DATA_PATH)
    valid_data = mydataset.DS(text_field, label_field, VALID_DATA_PATH)

    cache = ".vector_cache"
    if not os.path.exists(cache):
        os.mkdir(cache)
    vectors = Vectors(name='./data/glove.42B.300d.txt', cache=cache)

    text_field.build_vocab(train_data, valid_data, vectors=vectors)
    label_field.build_vocab(train_data, valid_data)

    print("Loading successfully !")

    return train_data, valid_data

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    text_field = data.Field(lower=True)
    label_field = data.Field(unk_token=None, pad_token=None)

    train_data, valid_data = load_data(text_field, label_field)

    args.vocab_size = len(text_field.vocab)
    args.target_size = len(label_field.vocab)
    args.weight_matrix = text_field.vocab.vectors
    #print(label_field.vocab.itos)

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    train_iter = data.Iterator(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    valid_iter = data.Iterator(dataset=valid_data, batch_size=args.batch_size, shuffle=True)

    esim = model.ESIM(args)

    print("Starting training...")
    train.train(train_iter, valid_iter, esim, args)
        