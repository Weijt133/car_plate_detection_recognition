import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
from pathlib import Path
import torch
import pathlib

# 1) Enable Tensor Core precision-performance tradeoff
torch.set_float32_matmul_precision('medium')
# 2) Allow WindowsPath deserialization (resolves checkpoint loading issue)
torch.serialization.add_safe_globals([pathlib.WindowsPath])

import pytorch_lightning as pl
from deep_utils import mkdir_incremental, CRNNModelTorch, get_logger, TorchUtils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import CRNNDataset
from settings import Config
from torch.nn import CTCLoss
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

torch.backends.cudnn.benchmark = True


class LitCRNN(pl.LightningModule):
    def __init__(self,
                 img_h, n_channels, n_classes,
                 n_hidden, lstm_input,
                 lr, lr_reduce_factor, lr_patience, min_lr,
                 label2char: dict):
        super().__init__()
        # Save all parameters to hparams (facilitates checkpoint restoration)
        self.save_hyperparameters()
        # Store the character mapping, used during decode_predictions
        self.label2char = label2char

        # Backbone network & loss function
        self.model = CRNNModelTorch(
            img_h=self.hparams.img_h,
            n_channels=self.hparams.n_channels,
            n_classes=self.hparams.n_classes,
            n_hidden=self.hparams.n_hidden,
            lstm_input=self.hparams.lstm_input
        )
        self.model.apply(self.model.weights_init)
        self.criterion = CTCLoss(reduction='mean')

    def forward(self, x):
        # x: [B, C, H, W] -> logits: [W', B, n_classes]
        logit = self.model(x)
        return torch.transpose(logit, 1, 0)

    def get_loss(self, batch):
        imgs, labels, lengths = batch
        lengths = lengths.squeeze(1)
        bs = imgs.size(0)
        # network output: [W', B, n_classes]
        logits = self.model(imgs)
        input_lengths = torch.LongTensor([logits.size(0)] * bs)
        loss = self.criterion(logits, labels, input_lengths, lengths)
        return loss, bs

    def training_step(self, batch, batch_idx):
        loss, bs = self.get_loss(batch)
        return {"loss": loss, "bs": bs}

    def training_epoch_end(self, outputs):
        avg = sum(o["loss"] for o in outputs) / sum(o["bs"] for o in outputs)
        self.log("train_loss", avg.item())

    def validation_step(self, batch, batch_idx):
        loss, bs = self.get_loss(batch)
        return {"loss": loss, "bs": bs}

    def validation_epoch_end(self, outputs):
        avg = sum(o["loss"] for o in outputs) / sum(o["bs"] for o in outputs)
        self.log("val_loss", avg.item())

    def test_step(self, batch, batch_idx):
        loss, bs = self.get_loss(batch)
        return {"loss": loss, "bs": bs}

    def test_epoch_end(self, outputs):
        avg = sum(o["loss"] for o in outputs) / sum(o["bs"] for o in outputs)
        self.log("test_loss", avg.item())

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        sch = ReduceLROnPlateau(
            opt,
            mode='min',
            factor=self.hparams.lr_reduce_factor,
            patience=self.hparams.lr_patience,
            verbose=True,
            min_lr=self.hparams.min_lr
        )
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val_loss"}

    @staticmethod
    def get_loaders(config):
        train_ds = CRNNDataset(
            root=config.train_directory,
            characters=config.alphabets,
            transform=config.train_transform
        )
        val_ds = CRNNDataset(
            root=config.val_directory,
            characters=config.alphabets,
            transform=config.val_transform
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.n_workers,
            collate_fn=train_ds.collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.n_workers,
            collate_fn=val_ds.collate_fn
        )
        return train_loader, val_loader

    def decode_predictions(self, preds):
        """
        Decode model outputs (logits) into strings:
        - preds: [W', B, n_classes]
        - Skip blank (0), merge repeated characters
        Returns List[str], length == B
        """
        # Convert to [B, W', n_classes]
        preds = preds.permute(1, 0, 2)
        idxs = preds.argmax(dim=2)
        texts = []
        for seq in idxs:
            last = None
            chars = []
            for idx in seq:
                i = idx.item()
                if i != 0 and i != last:
                    chars.append(self.label2char.get(i, ""))
                last = i
            texts.append("".join(chars))
        return texts


def main():
    parser = ArgumentParser()
    # parser.add_argument("--csv_file", type=Path,
    #                     default=r"D:\\444prj\\crnn-pytorch-master\\data-dir\\dataset.csv",
    #                     help="dataset")
    parser.add_argument("--train_directory", type=Path,
                        default=r"D:\\444prj\\crnn-pytorch-master\\data-dir\\train")
    parser.add_argument("--val_directory", type=Path,
                        default=r"D:\\444prj\\crnn-pytorch-master\\data-dir\\val")
    parser.add_argument("--output_dir", type=Path, default="./output")
    parser.add_argument("--epochs", type=int, default=300, help="Epoch_max")
    parser.add_argument("--device", default="cuda",
                        help="'cuda' or 'cpu'")
    parser.add_argument("--mean", nargs="+", type=float,
                        default=[0.4845])
    parser.add_argument("--std", nargs="+", type=float,
                        default=[0.1884])
    parser.add_argument("--img_w", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--alphabets", default='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_')
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    config = Config()
    config.update_config_param(args)

    output_dir = mkdir_incremental(str(config.output_dir))
    logger = get_logger("crnn-lightning", log_path=output_dir / "log.log")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience),
        ModelCheckpoint(dirpath=output_dir, filename=config.file_name, monitor="val_loss", verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    use_gpu = torch.cuda.is_available() and config.device == "cuda"
    trainer = pl.Trainer(
        accelerator='gpu' if use_gpu else 'cpu',
        devices=1 if use_gpu else None,
        max_epochs=config.epochs,
        min_epochs=max(1, config.epochs // 10),
        callbacks=callbacks,
        default_root_dir=output_dir
    )

    lit_crnn = LitCRNN(
        config.img_h, config.n_channels, config.n_classes,
        config.n_hidden, config.lstm_input,
        config.lr, config.lr_reduce_factor, config.lr_patience, config.min_lr,
        config.label2char
    )
    train_loader, val_loader = lit_crnn.get_loaders(config)

    # Begin training
    trainer.fit(lit_crnn, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Test
    trainer.test(lit_crnn, ckpt_path="best", dataloaders=val_loader)


if __name__ == '__main__':
    main()
