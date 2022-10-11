import pytorch_lightning as pl
from transformers import Adafactor
import torch
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd
from torch import nn

import deepspeed
import numpy as np
from Datasets import Custom_Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(args.embedding_dim, args.embedding_dim_2),
            nn.ReLU(),
            #nn.Linear(args.embedding_dim_2, args.embedding_dim_3),
            #nn.ReLU(),
            #nn.Linear(args.embedding_dim_3, args.embedding_dim_4)
        )

        self.decoder = nn.Sequential(
            #nn.Linear(args.embedding_dim_4, args.embedding_dim_3),
            #nn.ReLU(),
            #nn.Linear(args.embedding_dim_3, args.embedding_dim_2),
            #n.ReLU(),
            nn.Linear(args.embedding_dim_2, args.embedding_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

class AutoEncoder(pl.LightningModule):
    def __init__(self, hparams):
        super(AutoEncoder, self).__init__()     

        self.train_path = hparams.train_path
        self.eval_path = hparams.eval_path
        self.loss_function = nn.MSELoss()
        self.model = Network(hparams)
        self.save_hyperparameters(hparams)

        self.prediction_val = []
        self.actual_val = []
        self.prediction_train = []
        self.actual_train = []

        self.threshold = hparams.threshold

    def forward(self, input_vectors):
        reconstructed = self.model(input_vectors)
        
        target = input_vectors[:,-1]
        recon = reconstructed[:,-1]#.reshape(len(target), -1)
        """
        recon_final = torch.empty((len(target), 2), device=target.device)
        for i in range(len(target)):
            if recon[i] < 0.5:
                recon_final[i][0], recon_final[i][1] = 1, 0
            else:
                recon_final[i][0], recon_final[i][1] = 0, 1
        """

        return self.loss_function(recon, target), reconstructed
    
    def _step(self, batch):
        loss, reconstructed = self(
            input_vectors = batch['input_vectors'].to(torch.float16)
        )
        return loss, reconstructed

    def training_step(self, batch):
        loss, reconstructed = self._step(batch)
        for i in range(len(batch['input_vectors'])):
            self.actual_train.append(int(batch['input_vectors'][i][-1]))
            if reconstructed[i][-1] < self.threshold:
                self.prediction_train.append(0)
            else:
                self.prediction_train.append(1)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        acc = accuracy_score(self.actual_train, self.prediction_train)
        f1 = f1_score(self.actual_train, self.prediction_train, average='macro')
        precision = precision_score(self.actual_train, self.prediction_train)
        recall = recall_score(self.actual_train, self.prediction_train)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_precision', precision, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_recall', recall, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_actual_y=1', sum(self.actual_train), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_pred_y=1', sum(self.prediction_train), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.prediction_train = []
        self.actual_train = []

    def validation_step(self, batch, batch_idx):
        loss, reconstructed = self._step(batch)
        for i in range(len(batch['input_vectors'])):
            self.actual_val.append(int(batch['input_vectors'][i][-1]))
            if reconstructed[i][-1] < self.threshold:
                self.prediction_val.append(0)
            else:
                self.prediction_val.append(1)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def on_validation_epoch_end(self):
        acc = accuracy_score(self.actual_val, self.prediction_val)
        f1 = f1_score(self.actual_val, self.prediction_val, average='macro')
        precision = precision_score(self.actual_val, self.prediction_val)
        recall = recall_score(self.actual_val, self.prediction_val)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_precision', precision, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_actual_y=1', sum(self.actual_val), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_pred_y=1', sum(self.prediction_val), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.prediction_val = []
        self.actual_val = []

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
        train_dataset = Custom_Dataset(self.train_path)
        sampler = RandomSampler(train_dataset)
        dataloader = DataLoader(train_dataset,  sampler=sampler,batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        validation_dataset = Custom_Dataset(self.eval_path)
        return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)