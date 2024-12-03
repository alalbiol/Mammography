import torch
import torch.nn as nn
import torch.nn.functional as F



def get_loss(loss_name, **kwargs):
    """
    Get the loss class based on the loss name.
    
    Args:
        loss_name (str): The name of the loss.
    
    Returns:
        The loss class.
    """
    
    if "weight" in kwargs:
        kwargs["weight"] = torch.tensor(kwargs["weight"])
    
    if loss_name == "focal":
        print("Using Focal loss")
        print("kwargs: ", kwargs)
        return FocalLoss(**kwargs)
    
    if loss_name == "cross_entropy":
        print("Using CrossEntropy loss")
        print("kwargs: ", kwargs)
        return nn.CrossEntropyLoss(**kwargs)
    
    raise ValueError(f"Loss {loss_name} not found.")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Multiclass Focal Loss.
        
        Args:
            gamma (float): Focusing parameter gamma.
            alpha (list, float, or None): Weighting factor for each class.
                                         If None, no weighting is applied.
            reduction (str): Specifies the reduction to apply to the output: 
                             'none' | 'mean' | 'sum'. Default is 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Predicted logits (shape: [batch_size, num_classes]).
            targets (torch.Tensor): Ground truth labels (shape: [batch_size]).
        
        Returns:
            torch.Tensor: Computed focal loss.
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        # Get the probabilities of the target classes
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute the focal weight
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Apply alpha if specified
        if self.alpha is not None:
            alpha = torch.tensor(self.alpha, device=logits.device)
            alpha_factor = alpha.gather(0, targets)
            focal_weight *= alpha_factor
        
        # Compute the focal loss
        loss = -focal_weight * torch.log(target_probs)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class CrossEntropyMixup(nn.Module):
    def __init__(self, alpha=0.2, reduction='mean'):
        """
        Cross-entropy loss with mixup augmentation. In validation mode, the loss
        is computed as the cross-entropy loss without mixup.
        
        In mixup labels are converted to one-hot encodings and the loss is computed
        
        Args:
            alpha (float): Mixup interpolation parameter.
            reduction (str): Specifies the reduction to apply to the output: 
                             'none' | 'mean' | 'sum'. Default is 'mean'.
        """
        super(CrossEntropyMixup, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Predicted logits (shape: [batch_size, num_classes]).
            targets (torch.Tensor): Ground truth labels (shape: [batch_size]).
        
        Returns:
            torch.Tensor: Computed cross-entropy loss.
        """
        if self.training:
            # Mixup
            # Generate mixed inputs
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
            indices = torch.randperm(logits.size(0), device=logits.device)
            
            mixed_logits = lam * logits + (1 - lam) * logits[indices]
            logits = mixed_logits
            targets_onehot = F.one_hot(targets, num_classes=logits.size(1)).float()
            targets_mixup = self.alpha * targets_onehot + (1 - self.alpha) * targets_onehot[indices]
            loss = -torch.sum(targets_mixup * F.log_softmax(logits, dim=1), dim=1)
        else:
            # No mixup
            loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss