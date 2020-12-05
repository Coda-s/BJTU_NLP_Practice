import os
import json

import torchtext.data as data

class DS(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, data_path, **kwargs):
        examples = self.get_examlples(data_path, text_field, label_field)
        text_field.tokenize = lambda x: x.split()

        fields = {"sentence1" : text_field,
                  "sentence2" : text_field, 
                  "gold_label" : label_field}

        super(DS, self).__init__(examples, fields)

    def get_examlples(self, data_path, text_field, label_field):
        fields = {"sentence1" : ("sentence1", text_field),
                  "sentence2" : ("sentence2", text_field), 
                  "gold_label" : ("gold_label", label_field)}
        
        with open(data_path, "r") as f:
            datas = f.readlines()
            examples = [data.Example.fromJSON(dt, fields) for dt in datas]
        
        return examples







    