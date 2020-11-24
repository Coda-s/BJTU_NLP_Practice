import os
import torchtext.data as data


class DS(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, data_path, examples=None, **kwargs):
        with open(os.path.join(data_path, "seq.in"), "r") as f:
            text = f.readlines()
        with open(os.path.join(data_path, "seq.out"), "r") as f:
            label = f.readlines()
        
        fields = [("text", text_field), ("label", label_field)]
        examples = [data.Example.fromlist(dt, fields) for dt in zip(text, label)]
        text_field.tokenize = lambda x: x.split()
        super(DS, self).__init__(examples, fields)

    
    