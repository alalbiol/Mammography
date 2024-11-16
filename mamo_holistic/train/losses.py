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
