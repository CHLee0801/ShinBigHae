import pytorch_lightning as pl
from transformers import Adafactor
import torch
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch import nn
from torch.nn import functional as F

import deepspeed
from Datasets import Custom_Dataset
from sklearn.metrics import f1_score, accuracy_score

from .architectures import FC_Encoder, FC_Decoder

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        output_size = 512
        self.encoder = FC_Encoder(args.embedding_dim, args.embedding_dim_2)
        self.var = nn.Linear(output_size, args.embedding_dim)
        self.mu = nn.Linear(output_size, args.embedding_dim)

        self.decoder = FC_Decoder(args.embedding_dim)

    def encode(self, x):
        x = self.encoder(x)
        return self.mu(x), self.var(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VAE(pl.LightningModule):
    def __init__(self, hparams):
        super(VAE, self).__init__()     

        self.train_path = hparams.train_path
        self.eval_path = hparams.eval_path
        self.model = Network(hparams)
        self.save_hyperparameters(hparams)

        self.prediction = []
        self.actual = []

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def forward(self, input_vectors):
        reconstructed, mu, logvar = self.model(input_vectors)
        return self.loss_function(reconstructed, input_vectors, mu, logvar), reconstructed
    
    def _step(self, batch):
        loss, reconstructed = self(
            input_vectors = batch['input_vectors'].to(torch.float16)
        )
        return loss, reconstructed

    def training_step(self, batch):
        loss, _ = self._step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, reconstructed = self._step(batch)
        for i in range(len(batch['input_vectors'])):
            self.actual.append(int(batch['input_vectors'][i][-1]))
            if reconstructed[i][-1] < 0.5:
                self.prediction.append(0)
            else:
                self.prediction.append(1)

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
        train_dataset = Custom_Dataset(self.train_path)
        sampler = RandomSampler(train_dataset)
        dataloader = DataLoader(train_dataset,  sampler=sampler,batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        validation_dataset = Custom_Dataset(self.eval_path)
        return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)