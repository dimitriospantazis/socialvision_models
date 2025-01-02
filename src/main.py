# src/main.py

import sys
import os
import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
import torch
import argparse


# Append custom source directories to sys.path
sys.path.append('./src')
sys.path.append('./src/utils')

# Import custom modules
from config import config  # Import the config dictionary
from data import get_dataloaders
from model import LightningModel
from utils.training_utils import (
    download_best_checkpoint,
    log_best_model,
    setup_callbacks,
    setup_trainer,
    load_checkpoint,
    log_hyperparameters
)


def parse_arguments():
    """
    Parse command-line arguments and update the config dictionary.

    Returns:
        dict: Updated configuration dictionary.
    """
    parser = argparse.ArgumentParser(description="Train a video classifier.")
    # Dynamically add arguments based on the keys in the config dictionary
    for key, value in config.items():
        arg_type = type(value) if value is not None else str
        if isinstance(value, list):  # Handle list arguments
            parser.add_argument(f"--{key}", nargs="+", type=type(value[0]), default=value)
        elif isinstance(value, dict):  # Skip dictionaries for simplicity
            continue
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=value)
    
    args = parser.parse_args()
    return vars(args)


def main(train_tasks=None):
    """
    Main function to train the model, log artifacts to MLflow, and handle checkpoint downloads.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Update config with command-line arguments
    config.update(args)

    # Set matrix multiplication precision (optional)
    torch.set_float32_matmul_precision('high')  # Options: 'medium', 'high', 'highest'

    # 1. Prepare data
    train_loader, val_loader = get_dataloaders(config)

    # 2. Create the Lightning model
    pl_model = LightningModel(config)

    # 3. Handle checkpoint resumption
    resume_run_id = config.get("resume_run_id", None)
    if resume_run_id:
        print(f"Downloading best checkpoint from MLflow run_id: {resume_run_id}")
        checkpoint_path = download_best_checkpoint(resume_run_id, "best_model_ckpt")
        if checkpoint_path:
            load_checkpoint(pl_model, checkpoint_path)
            print(f"Checkpoint loaded from: {checkpoint_path}")
        else:
            print("Failed to download checkpoint. Starting training from scratch.")

    # 4. Set up MLflow Logger
    mlflow_logger = pl.loggers.MLFlowLogger(experiment_name=config["experiment_name"], run_name=config["run_name"])

    # 5. Instantiate callbacks
    callbacks = setup_callbacks(config)

    # 6. Create the Trainer
    trainer = setup_trainer(config, mlflow_logger, callbacks)

    # 7. Log hyperparameters
    log_hyperparameters(config, mlflow_logger)

    # 8. Train the model
    trainer.fit(pl_model, train_loader, val_loader)

    # 9. Retrieve the best checkpoint path
    best_ckpt_path = None
    for callback in callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            best_ckpt_path = callback.best_model_path
            break

    print(f"\nBest checkpoint saved at: {best_ckpt_path}")
    print(f"MLflow Run ID: {mlflow_logger.run_id}")

    if best_ckpt_path:
        # Load the best model from the checkpoint
        best_model = LightningModel.load_from_checkpoint(best_ckpt_path)

        # Log the best model and checkpoint to MLflow
        log_best_model(best_model, mlflow_logger.run_id, best_ckpt_path)
    else:
        print("No best checkpoint found to log.")

    # Optional: Log the last checkpoint if needed
    last_ckpt_path = os.path.join("model_checkpoints", "fine_tuned", "last.ckpt")
    if os.path.exists(last_ckpt_path):
        mlflow.log_artifact(
            local_path=last_ckpt_path,
            artifact_path="last_checkpoint",
            run_id=mlflow_logger.run_id
        )
    else:
        print("No last checkpoint found to log.")

if __name__ == "__main__":
    main()
