import os
import pandas

path = os.path.abspath(".")
path += "/data/train/pos/10006_7.txt"

with open(path, "r", encoding="UTF-8") as f:
    data = f.read()
    print(data)