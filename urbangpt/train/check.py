import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
print(curPath, rootPath)
sys.path.append(rootPath)


import json
import pickle
from torch.utils.data import DataLoader
import transformers
from train_st import LazySupervisedDataset_ST, make_supervised_stdata_module, DataCollatorForSupervisedDataset


# Define paths
data_path = '/home/zhangmin/toby/UrbanGPT/data/train_data/multi_NYC_sample.json'
st_data_path = '/home/zhangmin/toby/UrbanGPT/data/discharge/st_data.pkl'

# Initialize tokenizer
model_name_or_path = '/home/zhangmin/toby/UrbanGPT/checkpoints/vicuna-7b-v1.5-16k'  # Replace with your model name or path if different
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

# Define dummy data arguments


class DataArguments:
    def __init__(self):
        self.data_path = data_path
        self.lazy_preprocess = True
        self.is_st = True
        self.sep_st_conv_front = False
        self.st_token_len = 0
        self.st_content = None
        self.st_data_path = st_data_path
        self.use_st_start_end = False


data_args = DataArguments()

# Create dataset and data collator
data_module = make_supervised_stdata_module(tokenizer=tokenizer, data_args=data_args)
train_dataset = data_module['train_dataset']
data_collator = data_module['data_collator']

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=data_collator)

# Iterate through a few batches to verify the data
for batch in train_loader:
    print("Input IDs:", batch['input_ids'])
    print("Labels:", batch['labels'])
    print("ST Data X:", batch['st_data_x'])
    print("ST Data Y:", batch['st_data_y'])
    print("Region Start:", batch['region_start'])
    print("Region End:", batch['region_end'])


print("Dataset loaded and verified successfully.")
