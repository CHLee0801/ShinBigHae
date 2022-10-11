import pytorch_lightning as pl
from transformers import Adafactor, BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import functional as F

import pandas as pd
from torch import nn

import deepspeed
import numpy as np
from Datasets import Custom_Dataset_For_Bert
from sklearn.metrics import f1_score, accuracy_score


class BERT(pl.LightningModule):
    def __init__(self, hparams):
        super(BERT, self).__init__()    

        self.train_path = hparams.train_path
        self.eval_path = hparams.eval_path
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
        self.save_hyperparameters(hparams)

        self.prediction = []
        self.actual = []

    def forward(self, input_vectors, labels):
        outputs = self.model(input_vectors, labels=labels)
        return outputs['loss'], outputs['logits']
    
    def _step(self, batch):
        loss, logits = self(
            input_vectors = batch['source_ids'],
            labels = batch['labels']
        )
        return loss, logits

    def training_step(self, batch):
        loss, _ = self._step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self._step(batch)
        pred = torch.argmax(F.softmax(logits), dim=1)

        self.prediction += list(pred.cpu())
        self.actual += list(batch['labels'].cpu())

        acc = accuracy_score(self.actual, self.prediction)
        f1 = f1_score(self.actual, self.prediction, average='macro')

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        self.prediction = []
        self.actual = []

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        parameters = model.parameters()

        if self.hparams.accelerator=='deepspeed_stage_2':
            optimizer = deepspeed.ops.adam.FusedAdam(parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, betas=(0.9, 0.95))
        elif self.hparams.accelerator=='deepspeed_stage_2_offload':
            optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, betas=(0.9, 0.95))
        else: 
            optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False)
            #optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)

        return [optimizer]

    def train_dataloader(self):
        train_dataset = Custom_Dataset_For_Bert(self.train_path, self.tokenizer)
        sampler = RandomSampler(train_dataset)
        dataloader = DataLoader(train_dataset,  sampler=sampler,batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        validation_dataset = Custom_Dataset_For_Bert(self.eval_path, self.tokenizer)
        return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)