# Add src to Python's path
import sys
sys.path.append('./src')
sys.path.append('./src/utils')

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

##################################
# 1) Random Horizontal Flip Transform
##################################
class RandomHorizontalFlipVideo:
    """
    Randomly horizontally flip the entire video
    (all frames) with probability p.

    Expects video_tensor to have shape [C, T, H, W].
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video_tensor):
        if random.random() < self.p:
            # Flip the width dimension (last dimension)
            video_tensor = torch.flip(video_tensor, dims=[3])
        return video_tensor




##################################
# 2) VideoDataset (with optional transform)
##################################
class VideoDataset(Dataset):
    def __init__(self, video_dir, labels_csv, transform=None):
        """
        Args:
            video_dir (list or str): one or more directories where .pt files are stored
            labels_csv (str): path to the CSV containing label info
            transform (callable): optional transform (e.g. RandomHorizontalFlipVideo)
        """
        # If video_dir is just one path as a string, wrap it in a list
        if isinstance(video_dir, str):
            video_dir = [video_dir]

        self.video_dir = video_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform

        # Keep only the columns we need
        columns = [
            'video_name',
            'distance_score', 'distance_confidence',
            'object_score', 'object_confidence',
            'expanse_score', 'expanse_confidence',
            'facingness_score', 'facingness_confidence',
            'communicating_score', 'communicating_confidence',
            'joint_score', 'joint_confidence',
            'valence_score', 'valence_confidence',
            'arousal_score', 'arousal_confidence',
            'peoplecount', 'peoplecount_certain',
            'location_score', 'location_confidence'
        ]
        self.labels_df = self.labels_df[columns]

        # Encode location into 0,1
        location_map = {"Indoor": 0, "Outdoor": 1}
        self.labels_df['location_score'] = self.labels_df['location_score'].map(location_map)

        # Encode peoplecount 0, 1, 2, 3, many into class numbers 0, 1, 2, 3, 4
        peoplecount_map = {"0": 0, "1": 1, "2": 2, "3": 3, "many": 4}
        self.labels_df['peoplecount'] = self.labels_df['peoplecount'].map(peoplecount_map)

        # Convert peoplecount_certain column from bool to 0/1
        self.labels_df['peoplecount_certain'] = self.labels_df['peoplecount_certain'].astype(int)

        # Set video_name as index for easy lookup
        self.labels_df.set_index("video_name", inplace=True)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Get video name from the index
        video_name = self.labels_df.index[idx]

        # Randomly select one of the directories if multiple
        selected_dir = np.random.choice(self.video_dir)
        video_path = os.path.join(selected_dir, video_name[:-4] + ".pt")

        # Retrieve labels for the current video
        label_row = self.labels_df.loc[video_name]
        labels = label_row.to_dict()  # Convert the row to a dictionary

        # Load the video tensor (shape [C, T, H, W]) [channels, num_frames, height, width]
        if os.path.exists(video_path):
            video_tensor = torch.load(video_path, weights_only=True)
        else:
            raise FileNotFoundError(f"Video tensor not found for {video_name}")

        # Apply transform (e.g. random flip) if provided
        if self.transform is not None:
            video_tensor = self.transform(video_tensor)

        return video_tensor, labels




##################################
# 3) get_dataloaders
##################################
def get_dataloaders(config):
    # Create training dataset with transform
    train_dataset = VideoDataset(
        video_dir=config["video_dir"],
        labels_csv=config["labels_csv"],
        transform=RandomHorizontalFlipVideo(p=0.5)
    )

    # Create validation dataset without transform
    val_dataset = VideoDataset(
        video_dir=config["video_dir"],
        labels_csv=config["labels_csv"],
        transform=None
    )

    # Split the datasets accordingly
    dataset_size = len(train_dataset)
    train_size = int(config["train_ratio"] * dataset_size)
    val_size = dataset_size - train_size

    # Use a fixed generator seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Optionally reduce dataset size for debugging
    if config.get("reduce_dataset", False):
        train_subset = Subset(train_subset, range(2000))
        val_subset   = Subset(val_subset, range(1000))

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader


