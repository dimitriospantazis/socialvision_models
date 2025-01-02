import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from loss import loss_function
from models_MiT import load_model, MultiHeadLinear  # Adapt to your local imports

class LightningModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)  # Logs hyperparams in Lightning
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.include_confidence = config["include_confidence"]
        self.learning_rate = config["learning_rate"]
        self.train_tasks = config["train_tasks"]

        # Load your 3D ResNet model
        self.model = load_model('multi_resnet3d50')  
        self.model.last_linear = MultiHeadLinear(in_features=2048)

        # Initialize each task weight to 1.0
        # self.num_tasks = 10  # Assuming x tasks
        # self.task_weights = torch.nn.Parameter(torch.ones(self.num_tasks, requires_grad=False))
        
    def forward(self, x):
        """
        Forward pass. x is a batch of video data.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training loop step. Return the loss.
        """
        inputs, labels = batch
        inputs = inputs.float()
        labels = {k: v.float() for k, v in labels.items()}
        outputs = self.forward(inputs)
        task_loss, tasks = loss_function(outputs, labels, 
                             self.alpha, 
                             self.beta, 
                             self.include_confidence, self.train_tasks)
        loss = torch.mean(task_loss)

        # Log the loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True) 

        # Log the task losses, using the tasks as names
        for task, task_loss in zip(tasks, task_loss):
            self.log(f"train_loss-{task}", task_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            
        # Log the learning rate
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True, logger=True) 

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation loop step. Return the loss.
        """
        inputs, labels = batch
        inputs = inputs.float()
        labels = {k: v.float() for k, v in labels.items()}
        outputs = self.forward(inputs)
        task_loss, tasks = loss_function(outputs, labels, 
                             self.alpha, 
                             self.beta, 
                             self.include_confidence, self.train_tasks)
        loss = torch.mean(task_loss)

        # Log the loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # Log the task losses, using the tasks as names
        for task, task_loss in zip(tasks, task_loss):
            self.log(f"val_loss-{task}", task_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def configure_optimizers(self):
        """
        Define optimizer and (optionally) LR scheduler.
        """
        optimizer = Adam(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        # If using CosineAnnealingWarmRestarts
        if self.hparams["scheduler"]["type"] == "CosineAnnealingWarmRestarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, 
                **self.hparams["scheduler"]["params"]
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def on_train_epoch_end(self):
        # Retrieve metrics from the callback_metrics dictionary
        train_loss = self.trainer.callback_metrics.get("train_loss")
        val_loss = self.trainer.callback_metrics.get("val_loss")

        # Print them in the console (will not be overwritten)
        print(
            f"\nEpoch {self.current_epoch} (train) "
            f"| train_loss: {train_loss:.4f} "
            f"| val_loss: {val_loss:.4f}\n"
        )
