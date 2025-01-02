# src/utils/training_utils.py

import sys
import os
import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from pytorch_lightning import Callback


def set_requires_grad(module, requires_grad: bool):
    for param in module.parameters():
        param.requires_grad = requires_grad

class ProgressiveUnfreezeCallback(Callback):
    def __init__(self, config):
        super().__init__()
        self.unfreeze_schedule = config["unfreeze_schedule"]
        self.low_lr = config["low_lr"]
        self.post_unfreeze_epochs = config["post_unfreeze_epochs"]
        self.last_unfreeze_epoch = None

    def setup(self, trainer, pl_module, stage=None):
        """
        Called when fit begins.
        """
        # Initially freeze all layers
        set_requires_grad(pl_module.model, False)

        # Unfreeze layers scheduled at epoch 0
        for layer_name, threshold in self.unfreeze_schedule.items():
            if threshold <= 0:
                self._unfreeze_layer(pl_module, layer_name)
        
    def on_train_epoch_start(self, trainer, pl_module):
        """
        Called at the start of each training epoch.
        """
        epoch = trainer.current_epoch
        model = pl_module.model
        layer_map = self._get_layer_map(model)

        layer_unfrozen = False

        # Iterate through the unfreeze schedule
        for layer_name, threshold in self.unfreeze_schedule.items():
            if epoch >= threshold:
                module = layer_map.get(layer_name, None)
                if module and not any(p.requires_grad for p in module.parameters()):
                    self._unfreeze_layer(pl_module, layer_name)
                    layer_unfrozen = True

        if layer_unfrozen:
            self.last_unfreeze_epoch = epoch
            # Rebuild optimizer & scheduler to include newly unfrozen params
            # Note: Depending on the PyTorch Lightning version, you might need to adjust this
            trainer.strategy.setup_optimizers(trainer)

        # Adjust learning rates during cooldown period
        if self.last_unfreeze_epoch is not None and (epoch - self.last_unfreeze_epoch) < self.post_unfreeze_epochs:
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.low_lr

    def _get_layer_map(self, model):
        """
        Create a mapping from layer names to actual modules.
        Supports nested modules like 'layer4.0'.
        """
        layer_map = {}
        for name, module in model.named_modules():
            if not name:
                continue  # Skip the root module with empty name
            if '.' not in name:
                try:
                    layer_map[name] = getattr(model, name)
                except AttributeError:
                    print(f"Warning: Model has no attribute named '{name}'")
            else:
                parent, child = name.split('.', 1)
                if parent in layer_map:
                    sub_module = layer_map[parent]
                    # Navigate deeper if needed
                    for sub_name in child.split('.'):
                        try:
                            sub_module = getattr(sub_module, sub_name)
                        except AttributeError:
                            sub_module = None
                            print(f"Warning: Submodule '{sub_name}' not found in '{parent}'")
                            break
                    if sub_module:
                        # Use full name for unique identification
                        layer_map[name] = sub_module
        return layer_map

    def _unfreeze_layer(self, pl_module, layer_name):
        """
        Unfreeze a specific layer by name.
        """
        model = pl_module.model
        layer_map = self._get_layer_map(model)
        module = layer_map.get(layer_name, None)
        if module:
            set_requires_grad(module, True)
            print(f"Unfroze layer: {layer_name}")
        else:
            print(f"Layer {layer_name} not found in the model.")


def download_best_checkpoint(run_id, artifact_path="best_model_ckpt"):
    """
    Downloads the best model checkpoint from MLflow artifacts.

    Args:
        run_id (str): The MLflow run ID.
        artifact_path (str): The path within the run's artifacts where the checkpoint is stored.

    Returns:
        str: Local path to the downloaded checkpoint file or None if failed.
    """
    try:
        # Download artifacts from the specified run and artifact path
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        
        # Check if the artifact is a file
        if os.path.isfile(local_path) and local_path.endswith(".ckpt"):
            return local_path
        elif os.path.isdir(local_path):
            # If it's a directory, search for the .ckpt file within
            checkpoint_files = [f for f in os.listdir(local_path) if f.endswith(".ckpt")]
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint file found in artifact path: {artifact_path}")
            # Return the full path to the first .ckpt file found
            checkpoint_path = os.path.join(local_path, checkpoint_files[0])
            return checkpoint_path
        else:
            raise FileNotFoundError(f"No checkpoint file found in artifact path: {artifact_path}")
    except Exception as e:
        print(f"Error downloading checkpoint from MLflow: {e}")
        return None

def log_best_model(best_model, run_id, checkpoint_path):
    """
    Logs the best model and its checkpoint to MLflow.

    Args:
        best_model (pl.LightningModule): The best PyTorch Lightning model.
        run_id (str): The MLflow run ID.
        checkpoint_path (str): Path to the best checkpoint file.
    """
    mlflow.pytorch.log_model(
        pytorch_model=best_model,
        artifact_path="best_model",
        run_id=run_id
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        mlflow.log_artifact(
            local_path=checkpoint_path,
            artifact_path="best_model_ckpt",
            run_id=run_id
        )
    else:
        print("No best checkpoint found to log.")


def setup_callbacks(config):
    """
    Sets up the callbacks for the Trainer.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        list: List of instantiated callbacks.
    """
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("model_checkpoints", "fine_tuned"),
        filename="model_{epoch}_{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,       # Only save the best checkpoint
        save_last=True,     # Optionally save the last checkpoint
        every_n_epochs=1,
        save_weights_only=True
    )

    early_stopping_config = config.get("early_stopping", {})
    early_stopping_callback = EarlyStopping(
        monitor=early_stopping_config.get("monitor", "val_loss"),
        patience=early_stopping_config.get("patience", 3),
        mode=early_stopping_config.get("mode", "min"),
        verbose=early_stopping_config.get("verbose", True)
    )

    callbacks = [
        ProgressiveUnfreezeCallback(config),
        checkpoint_callback,
        early_stopping_callback,
    ]
    return callbacks

def setup_trainer(config, logger, callbacks):
    """
    Sets up the PyTorch Lightning Trainer.

    Args:
        config (dict): Configuration dictionary.
        logger (pl.loggers.Logger): Logger instance.
        callbacks (list): List of callbacks.

    Returns:
        pl.Trainer: Configured Trainer.
    """
    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=config["validation_interval"],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        enable_model_summary=False
    )
    return trainer

def load_checkpoint(model, checkpoint_path):
    """
    Loads the model state from a checkpoint.

    Args:
        model (pl.LightningModule): The PyTorch Lightning model.
        checkpoint_path (str): Path to the checkpoint file.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")

def log_hyperparameters(config, logger):
    """
    Logs hyperparameters to the MLflow run via the logger's experiment.

    Args:
        config (dict): Configuration dictionary.
        logger (pl.loggers.Logger): Logger instance.
    """
    for k, v in config.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                logger.experiment.log_param(
                    run_id=logger.run_id,
                    key=f"{k}.{sub_k}",
                    value=sub_v
                )
        else:
            logger.experiment.log_param(
                run_id=logger.run_id,
                key=k,
                value=v
            )
