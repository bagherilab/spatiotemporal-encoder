import torch
import torch.nn.functional as F


def edge_aware_mse_loss(pred, target, edge_threshold=0.01, edge_multiplier=50.0):
    """
    Custom MSE loss that applies a multiplier to edge pixels.
    
    An edge pixel is defined as a non-black pixel that has at least one black neighbor
    but isn't completely surrounded by black pixels on all 4 sides.
    
    Args:
        pred: Predicted image tensor (B x C x H x W)
        target: Target image tensor (B x C x H x W)
        edge_threshold: Threshold to consider a pixel as black (default: 0.01)
        edge_multiplier: Multiplier for edge pixels (default: 50.0)
        
    Returns:
        Loss value
    """
    # Calculate standard MSE loss
    mse_loss = F.mse_loss(pred, target, reduction='none')
    
    # Create a mask for black pixels (assuming input is in [0,1] range)
    is_black = (target <= edge_threshold).float()
    
    # Pad the black mask for edge detection
    padded_black = F.pad(is_black, (1, 1, 1, 1), mode='constant', value=0)
    
    # Check 4-connected neighbors (up, down, left, right)
    neighbors = torch.zeros_like(is_black)
    neighbors += padded_black[:, :, 2:, 1:-1]  # down
    neighbors += padded_black[:, :, :-2, 1:-1]  # up
    neighbors += padded_black[:, :, 1:-1, 2:]   # right
    neighbors += padded_black[:, :, 1:-1, :-2]  # left
    
    # A pixel is an edge if:
    # 1. It's not black itself
    # 2. It has at least one black neighbor
    # 3. It's not completely surrounded by black pixels
    is_edge = ((1 - is_black) * 
              (neighbors > 0).float() * 
              (neighbors < 4).float())
    
    # Create weight tensor
    weights = torch.ones_like(mse_loss)
    weights[is_edge.bool()] = edge_multiplier
    
    # Apply weights and return mean loss
    return (mse_loss * weights).mean()