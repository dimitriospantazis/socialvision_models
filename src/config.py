# Configuration for Training Video Classifier
config = {
    # -----------------------------
    # Experiment Settings
    # -----------------------------
    "experiment_name": "train_video_classifier",  # Identifier for the experiment (e.g. 'train_video_classifier')
    "run_name": "tmp-video_classifier_2_layer4.2-joint-dropoutH",               # Name for this run (e.g. 'video_classifier_1_lastlayer')
    "resume_run_id": '15da69f6c12f4ecf84c49257fa1e9607',     # Run ID to resume training (None to start fresh)
    "train_tasks": ['joint'],                        # Task to train on (e.g. 'joint', 'communicating', 'valence', 'all')
    "num_epochs": 30,                             # Total number of training epochs
    "validation_interval": 1,                     # Number of epochs between validation checks

    # -----------------------------
    # Data Settings
    # -----------------------------
    "num_workers": 8,                             # Number of data loading workers
    "reduce_dataset": False,                       # Flag to reduce dataset size for experiments
    "video_dir": ["H:\\tmp\\processed_videos"],   # List of directories containing videos
    "labels_csv": "data/annotations/video_annotations_combined_43392.csv",  # Path to labels CSV
    "train_ratio": 0.9,                           # Proportion of data for training
    "batch_size": 24,                             # Number of samples per batch

    # -----------------------------
    # Model Hyperparameters
    # -----------------------------
    "learning_rate": 0.000005,                     # Initial learning rate
    "unfreeze_schedule": {                        # Epochs to unfreeze specific layers
        "conv1": 200,
        "bn1": 200,
        "relu": 200,
        "layer1": 200,
        "layer2": 200,
        "layer3": 200,
        "layer4.0": 200,  # Unfreeze first Bottleneck block at epoch 100
        "layer4.1": 200,  # Unfreeze second Bottleneck block at epoch 150
        "layer4.2": 0,  # Unfreeze third Bottleneck block at epoch 200
        "last_linear": 0,
    },

    # -----------------------------
    # Loss Function Parameters
    # -----------------------------
    "alpha": 0,                                   # Weight for loss component A
    "beta": 1.0,                                  # Weight for loss component B
    "include_confidence": False,                  # Include confidence in loss calculation

    # -----------------------------
    # Learning Rate Scheduler
    # -----------------------------
    "scheduler": {
        "type": "CosineAnnealingWarmRestarts",    # Scheduler type
        "params": {                               # Scheduler parameters
            "T_0": 10,                             # Initial period for restarts
            "T_mult": 2,                           # Multiplicative factor for period
            "eta_min": 1.0e-6,                     # Minimum learning rate
        },
    },
    "low_lr": 1e-5,                               # Lower bound for learning rate immediately after unfreezing
    "post_unfreeze_epochs": 0,                    # Epochs to limit lr (to low_lr) after unfreezing (e.g. 2)

    # -----------------------------
    # Early Stopping
    # -----------------------------
    "early_stopping": {                          
        "monitor": "val_loss",                    # Metric to monitor
        "patience": 4,                            # Epochs to wait for improvement
        "mode": "min",                            # Mode: 'min' for loss, 'max' for accuracy
        "verbose": True,                          # Print messages on stopping
    },
}
