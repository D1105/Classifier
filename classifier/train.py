import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
import classifier
import wandb

wandb_logger = WandbLogger(log_model="all")
dm = classifier.data.CIFAR10DataModule(data_dir="dataset/", batch_size=classifier.batch_size, num_workers=4)
trainer = L.Trainer(logger = wandb_logger, accelerator = "auto", devices = 1, min_epochs = 1, max_epochs = classifier.epochs)