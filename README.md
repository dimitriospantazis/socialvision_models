# Training ResNet3D for Social Vision Interactions in Moments in Time Videos

This project trains a ResNet3D model to learn social vision interactions in videos from the Moments in Time dataset. It builds on the annotations generated by [socialvision_GPT_annotations](https://github.com/dimitriospantazis/socialvision_GPT_annotations) and leverages PyTorch Lightning and MLflow for efficient model training and result logging.

## Project Overview

### Purpose
The goal of this project is to develop a deep learning model that can understand and predict social interactions in videos. The annotations derived from [socialvision_GPT_annotations](https://github.com/dimitriospantazis/socialvision_GPT_annotations) serve as training labels, enabling the model to learn nuanced social dynamics such as:

- Proximity between individuals
- Object interaction vs. person-to-person interaction
- Spatial expanse of the scene
- Facingness (degree of mutual orientation)
- Communication (verbal or non-verbal)
- Joint action and coordination
- Emotional valence and arousal

### Approach
The training pipeline involves adapting and extending the ResNet3D architecture, incorporating a custom loss function, and optimizing the model using MSE. The resulting model can be used for downstream tasks such as video classification, human behavior analysis, and multimedia content understanding.

## How It Works

### Key Components
1. **Training Script (`main.py`)**: Orchestrates the training process using PyTorch Lightning. Includes MLflow integration for logging results and model parameters.
2. **Lightning Model (`model.py`)**: Defines the PyTorch Lightning module for ResNet3D.
3. **Modified MiT Model (`model_MiT.py`)**: Contains the adapted ResNet3D architecture from the original Moments in Time dataset with a new classification head and dropout layers.
4. **Custom Loss Function (`loss.py`)**: Implements the loss function used to optimize the model, focusing on Mean Squared Error (MSE).
5. **Dataset and Data Loaders (`data.py`)**: Prepares the video dataset and data loaders to feed the model during training. Videos and their corresponding annotations are processed for training and validation.

### Data and Annotations
The video dataset is derived from the Moments in Time collection, while the training labels are generated using the annotation pipeline from the [socialvision_GPT_annotations](https://github.com/dimitriospantazis/socialvision_GPT_annotations) project. The labels include detailed metadata about social interactions and scene dynamics.

## Implementation

### Training Workflow
1. **Dataset Preparation**: Videos are loaded, preprocessed, and paired with their respective annotations.
2. **Model Training**: The adapted ResNet3D model is trained using PyTorch Lightning. The training process includes logging intermediate results to MLflow for performance monitoring.
3. **Result Logging**: Metrics, hyperparameters, and model artifacts are logged in MLflow for reproducibility and analysis.

### Logging with MLflow
MLflow is used to track:
- Model parameters
- Training and validation metrics
- Loss curves
- Final model checkpoints

## Dependencies
- PyTorch Lightning
- MLflow
- PyTorch
- Moments in Time dataset
- [socialvision_GPT_annotations](https://github.com/dimitriospantazis/socialvision_GPT_annotations)

## Getting Started

### Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/your_username/your_project_name.git
cd your_project_name
pip install -r requirements.txt

## Usage
1. **Prepare the video dataset and annotations**: Follow the instructions in [socialvision_GPT_annotations](https://github.com/dimitriospantazis/socialvision_GPT_annotations).
2. **Update the paths**: Modify the paths to the dataset and annotation files in `data.py`.
3. **Train the model**:
   ```bash
   python main.py
4. **Monitor the training process and results using MLflow:
   ```bash
   mlflow ui 

## Output
-Model checkpoints
-Training and validation logs
-MLflow tracking and visualization

## Benefits
This project bridges the gap between automated video annotation and deep learning, creating a robust pipeline for analyzing social interactions in videos. It enables detailed insights into human behavior and environmental contexts, supporting research in social vision and multimedia analysis.




