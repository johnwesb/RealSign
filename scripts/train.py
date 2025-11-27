#!/usr/bin/env python
"""
Training script for ISL model.
Uses PyTorch Lightning for simplicity (add to requirements if needed).
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import DictConfig
from data.dataset import get_dataloader
from src.utils import load_config

class ISLClassifier(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=42,
            hidden_size=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            batch_first=True
        )
        self.fc = nn.Linear(config.model.hidden_dim, config.model.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        acc = (pred.argmax(1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)
        return optimizer

def main():
    parser = argparse.ArgumentParser(description="Train RealSign model")
    parser.add_argument("--config", type=str, default="configs/isl_train.yaml", help="Config file")
    parser.add_argument("--output-dir", type=str, default="models/", help="Output dir")
    args = parser.parse_args()

    config = load_config(args.config)

    train_loader = get_dataloader(config.data.train_path, config.data.batch_size)
    val_loader = get_dataloader(config.data.val_path, config.data.batch_size)

    model = ISLClassifier(config)

    checkpoint = ModelCheckpoint(dirpath=args.output_dir, filename="isl-{epoch:02d}-{val_acc:.2f}")

    trainer = Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        callbacks=[checkpoint],
        default_root_dir=config.trainer.log_dir
    )

    trainer.fit(model, train_loader, val_loader)

    # Export to ONNX
    dummy_input = torch.randn(1, config.data.sequence_length, 42)
    torch.onnx.export(model, dummy_input, f"{args.output_dir}/isl_latest.onnx")

if __name__ == "__main__":
    main()