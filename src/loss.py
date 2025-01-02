import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_score_confidence_loss(outputs, labels, question_type, alpha=1, include_confidence = True):

    # Get keys for the question type
    score_key = f"{question_type}_score"
    confidence_key = f"{question_type}_confidence"

    # Extract scores and confidences from the model's outputs and flatten them
    scores_pred = outputs[score_key].flatten()
    confidences_pred = outputs[confidence_key].flatten()

    # Extract target scores and confidences from the labels and flatten them
    scores_target = labels[score_key].flatten()
    confidences_target = labels[confidence_key].flatten()

    # If 'location', apply binary cross entropy loss instead of MSE
    if question_type == "location":
        criterion = nn.BCEWithLogitsLoss(reduction="none") # Already applies sigmoid to scores
    else:
        #criterion = nn.L1Loss(reduction="none") # Element-wise L1 loss
        criterion = nn.MSELoss(reduction="none") # Element-wise MSE loss
        scores_pred = torch.sigmoid(scores_pred) # Apply sigmoid to scores


    # Create a mask for valid entries (scores_target, but we assume that confidences_target does not have NaN)
    valid_mask = ~torch.isnan(scores_target) 

    # Compute individual losses weighted by the label confidences
    if torch.any(valid_mask):  # Ensure there's at least one valid target
        if include_confidence:
            score_loss = torch.dot(confidences_target[valid_mask], criterion(scores_pred[valid_mask], scores_target[valid_mask])) / len(scores_target[valid_mask])
        else:
            score_loss = torch.sum(criterion(scores_pred[valid_mask], scores_target[valid_mask]) / len(scores_target[valid_mask]))
    else:
        score_loss = 0.0

    # Compute the confidence loss (no masking needed because we assume no NaN values, they have already been replaced with 0)
    if include_confidence:
        confidence_loss = torch.mean(criterion(confidences_pred, confidences_target))
    else:
        confidence_loss = 0.0

    # Compute and return the combined loss
    combined_loss = score_loss + alpha * confidence_loss

    return combined_loss

def compute_peoplecount_loss(outputs, labels, beta=1):
    
    # Extract people count logits from the model's outputs
    peoplecount_pred = outputs["peoplecount"]
    peoplecount_certain_pred = outputs["peoplecount_certain"].flatten()

    # Extract target people count labels from the labels
    peoplecount_target = labels["peoplecount"]
    peoplecount_certain_target = labels["peoplecount_certain"]

    # Create mask for valid targets (non-NaN values). Include both peoplecount and peoplecount_certain, so the peoplecount_loss is only computed only for certain targets
    valid_mask = ~torch.isnan(peoplecount_target) & ~torch.isnan(peoplecount_certain_target)

    # Compute loss
    if torch.any(valid_mask):  # Ensure there's at least one valid target
        peoplecount_loss = F.cross_entropy(peoplecount_pred[valid_mask], peoplecount_target[valid_mask].long()) # Compute cross entropy, averaged across valid targets. No need to weight by "certain" since it's a categorical loss
    else:
        peoplecount_loss = 0.0

    # Compute the confidence loss for people count. Force BCEWithLogitsLoss to treat the target as a float tensor
    peoplecount_confidence_loss = torch.mean(nn.BCEWithLogitsLoss()(peoplecount_certain_pred, peoplecount_certain_target))

    # Compute and return the combined loss
    combined_loss = peoplecount_loss + beta * peoplecount_confidence_loss
    return combined_loss


def loss_function(outputs, labels, alpha=1.0, beta=1.0, include_confidence=True, train_tasks='all'):
    """
    Computes individual losses for each task and stacks them into a single tensor.

    Args:
        outputs (dict): Model outputs containing predictions for each task.
        labels (dict): Ground truth labels for each task.
        alpha (float): Weight for score-confidence losses.
        beta (float): Weight for peoplecount loss.
        include_confidence (bool): Whether to include confidence in score-confidence losses.

    Returns:
        torch.Tensor: Stacked individual losses.
        list: List of task names corresponding to each loss.
    """
    losses = []
    tasks = []

    ####################################
    # Score-Confidence Based Losses
    ####################################

    # Distance Loss
    if train_tasks == 'all' or 'distance' in train_tasks:
        losses.append(compute_score_confidence_loss(outputs, labels, "distance", alpha, include_confidence))
        tasks.append("distance")

    # Object Loss
    if train_tasks == 'all' or 'object' in train_tasks:
        losses.append(compute_score_confidence_loss(outputs, labels, "object", alpha, include_confidence))
        tasks.append("object")

    # Expanse Loss
    if train_tasks == 'all' or 'expanse' in train_tasks:
        losses.append(compute_score_confidence_loss(outputs, labels, "expanse", alpha, include_confidence))
        tasks.append("expanse")

    # Facingness Loss
    if train_tasks == 'all' or 'facingness' in train_tasks:
        losses.append(compute_score_confidence_loss(outputs, labels, "facingness", alpha, include_confidence))
        tasks.append("facingness")

    # Communicating Loss
    if train_tasks == 'all' or 'communicating' in train_tasks:
        losses.append(compute_score_confidence_loss(outputs, labels, "communicating", alpha, include_confidence))
        tasks.append("communicating")

    # Joint Loss
    if train_tasks == 'all' or 'joint' in train_tasks:
        losses.append(compute_score_confidence_loss(outputs, labels, "joint", alpha, include_confidence))
        tasks.append("joint")

    # Valence Loss
    if train_tasks == 'all' or 'valence' in train_tasks:
        losses.append(compute_score_confidence_loss(outputs, labels, "valence", alpha, include_confidence))
        tasks.append("valence")
        
    # Arousal Loss
    if train_tasks == 'all' or 'arousal' in train_tasks:
        losses.append(compute_score_confidence_loss(outputs, labels, "arousal", alpha, include_confidence))
        tasks.append("arousal")

    # Location Loss
    if train_tasks == 'all' or 'location' in train_tasks:
        # Scale the location task (0.02 good for other tasks, 0.2 good for location task)
        losses.append(compute_score_confidence_loss(outputs, labels, "location", alpha, include_confidence)  * (0.02 / 0.2)) 
        tasks.append("location")

    ####################################
    # Peoplecount Loss
    ####################################

    # Peoplecount Loss
    if train_tasks == 'all' or 'peoplecount' in train_tasks:
        # Scale the peoplecount task (0.02 good for other tasks, 0.5 good for peoplecount task)
        losses.append(compute_peoplecount_loss(outputs, labels, beta)  * (0.02 / 0.5)) 
        tasks.append("peoplecount")

    # Stack all individual losses into a single tensor
    stacked_losses = torch.stack(losses)

    return stacked_losses, tasks



