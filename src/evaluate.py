import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import sys
# Append custom source directories to sys.path
sys.path.append('./src')
sys.path.append('./src/utils')

from loss import loss_function
from models_MiT import load_model, MultiHeadLinear  # Adapt to your local imports
from utils.training_utils import load_checkpoint
from data import get_dataloaders
from config import config  # Import the config dictionary

train_loader, val_loader = get_dataloaders(config)

checkpoint_path = os.path.join(os.getcwd(),'model_checkpoints','fine_tuned','model_epoch=6_val_loss=0.0211.ckpt')

# Load 3D ResNet model
model = load_model('multi_resnet3d50')  
model.last_linear = MultiHeadLinear(in_features=2048)
load_checkpoint(model, checkpoint_path)
model.to('cuda')



myiter = iter(train_loader)
inputs, labels = next(myiter)
device = 0
inputs, labels = inputs.to(device), {k: v.to(device) for k, v in labels.items()}  # Move to device
outputs = model(inputs)

loss = loss_function(outputs, labels, alpha=0, beta=0, include_confidence=False, train_tasks = 'arousal')  # Calculate loss
print(loss)

task = 'arousal'
target = labels[f'{task}_score'].float()
#pred = torch.sigmoid(outputs[f'{question_type}_score']).flatten()
pred = outputs[f'{task}_score']
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


