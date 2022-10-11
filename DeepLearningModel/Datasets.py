from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset

class Custom_Dataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(dataset_path, encoding='utf-8')
        print(f'Getting dataset {dataset_path} with length {len(self.dataset)}')
        
    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):  
        return torch.from_numpy(example_batch.to_numpy())

    def __getitem__(self, index):
        input_vectors = self.convert_to_features(self.dataset.iloc[index], index=index) 
        return {"input_vectors": input_vectors}

class Custom_Dataset_For_Bert(Dataset):
    def __init__(self, dataset_path, tokenizer):
        self.dataset_path = dataset_path
        self.input_length = 128
        self.tokenizer = tokenizer
        self.dataset = pd.read_csv(dataset_path, encoding='utf-8')
        print(f'Getting dataset {dataset_path} with length {len(self.dataset)}')
        
    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):

        input_ = example_batch['input']
        targets = example_batch['output']

        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")
                    
        return source, targets

    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset.iloc[index], index=index) 

        source_ids = source["input_ids"].squeeze()
        src_mask    = source["attention_mask"].squeeze()
        
        return {"source_ids": source_ids, "source_mask": src_mask, "labels": targets}
