# Add src to Python's path
import sys
sys.path.append('./src')
sys.path.append('./src/utils')

import os
from video_dataset import VideoDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import models
from models import ResNet3D
import torch.optim as optim
from tqdm import tqdm
from loss import loss_function
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Load checkpoint (Optional)
def load_checkpoint(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    return model


# 1. Set up configurations (optional config file)
config = {
    "num_epochs": 200,
    "learning_rate": 0.001,
    "batch_size": 16,
    "print_interval": 400,
    "alpha": 1.0,
    "beta": 1.0,
    "num_workers": 1,
    "reduce_dataset": True
}

# Use config settings in your code
num_epochs = config["num_epochs"]
learning_rate = config["learning_rate"]
batch_size = config["batch_size"]
print_interval = config["print_interval"]
alpha = config["alpha"]
beta = config["beta"]
num_workers = config["num_workers"]


# 2. Load dataset and DataLoader
video_dir = [r'H:\tmp\processed_videos']
           

labels_csv = os.path.join(os.getcwd(), 'data','annotations','video_annotations_combined_43392.csv')
dataset =  VideoDataset(video_dir, labels_csv)

# Set the split ratio
train_ratio = 0.9
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

# Set reduce the dataset size for testing
if config["reduce_dataset"]:
    train_size = 2000
    val_size = 1000
    train_dataset = torch.utils.data.Subset(train_dataset, list(range(train_size)))
    val_dataset = torch.utils.data.Subset(val_dataset, list(range(val_size)))

# Define DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)





# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Initialize the model
model = models.load_model('multi_resnet3d50')
model.last_linear = models.MultiHeadLinear(in_features=2048) # Adjust final layer

# Move model to the device
model = model.to(device)



model_path = os.path.join(os.getcwd(), 'model_checkpoints','fine_tuned','reduced_dataset_2000_fulltrained','model_checkpoint_epoch_200.pt')
model = load_checkpoint(model_path)



model.eval()
val_loss = 0.0
with torch.no_grad():
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), {k: v.to(device) for k, v in labels.items()}  # Move to device
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        val_loss += loss.item()




# Load checkpoint (Optional)
def load_checkpoint(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    return model





checkpoint_path = 'model_checkpoint_epoch_7.pt'
model = load_checkpoint(checkpoint_path)


myiter = iter(train_loader)
inputs, labels = next(myiter)
inputs, labels = inputs.to(device), {k: v.to(device) for k, v in labels.items()}  # Move to device
outputs = model(inputs)

l = loss = loss_function(outputs, labels, alpha, beta, include_confidence)  # Calculate loss
print(l)

question_type = 'expanse'
target = labels[f'{question_type}_score'].float()
#pred = torch.sigmoid(outputs[f'{question_type}_score']).flatten()
pred = outputs[f'{question_type}_score']
target = target.unsqueeze(1)
#pred = pred.unsqueeze(1)
pred = torch.sigmoid(pred)
combined = torch.cat((target, pred), dim=1)
print(combined)



x1 = torch.tensor([0.5135, 0.4125, 0.2859, 0.5240, 0.8663, 0.1197, 0.2423, 0.6271, 0.8686,
        0.6988, 0.6787, 0.8489, 0.3349, 0.5492, 0.3574, 0.0928, 0.2274, 0.4609,
        0.8237, 0.4941, 0.6569, 0.1323, 0.3595])

x2 = torch.tensor([0.2000, 0.3000, 0.3000, 0.8000, 1.0000, 0.2000, 0.1000, 0.7000, 0.9000,
        0.7000, 0.8000, 0.9000, 0.4000, 0.9000, 0.1000, 0.2000, 0.2000, 0.7000,
        0.7000, 0.4000, 0.5000, 0.2000, 0.1000])

loss = nn.MSELoss()  # Element-wise MSE loss
mask = ~torch.isnan(target)
print(loss(target[mask].flatten(),pred[mask].flatten()))

#careful!
print(loss(target[mask],pred[mask]))


# Model not expressive enough
# Try much reduced dataset
# How large is the model?
# First train last layer, then rest. Or progressivly unfreeze layers


