import torch

def mixup_data(x, y, alpha=1.0):
    """
    Applies Mixup augmentation to the data.
    
    Args:
        x (Tensor): Input images.
        y (Tensor): Input labels (one-hot encoded or class indices).
        alpha (float): Parameter for the beta distribution.
        
    Returns:
        mixed_x (Tensor): Mixed images.
        y_a (Tensor): Labels of first set of images.
        y_b (Tensor): Labels of second set of images.
        lam (float): Mixup ratio.
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)   
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam