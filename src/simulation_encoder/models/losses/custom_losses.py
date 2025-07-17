import torch
import torch.nn.functional as F


def edge_aware_mse_loss(pred, target, black_threshold=0.01, edge_multiplier=100.0):
    """
    Custom MSE loss that applies a multiplier to edge pixels.
    
    An edge pixel is defined as a non-black pixel that is adjacent to at least
    one black pixel but isn't completely surrounded by black pixels.
    
    Args:
        pred: Predicted image tensor (B x C x H x W)
        target: Target image tensor (B x C x H x W)
        black_threshold: Threshold below which a pixel is considered black (default: 0.01)
        edge_multiplier: Multiplier for edge pixels (default: 50.0)
        
    Returns:
        Loss value
    """
    # Calculate standard MSE loss per channel
    mse_loss = F.mse_loss(pred, target, reduction='none')  # (B, C, H, W)
    
    # Calculate intensity (mean across channels)
    intensity = target.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    
    # Create a mask for black pixels (intensity < threshold)
    is_black = (intensity <= black_threshold).float()
    
    # Pad the black mask for edge detection
    padded_black = F.pad(is_black, (1, 1, 1, 1), mode='constant', value=0)
    
    # Check 4-connected neighbors (up, down, left, right)
    neighbors = torch.zeros_like(is_black)
    neighbors += padded_black[:, :, 2:, 1:-1]   # down
    neighbors += padded_black[:, :, :-2, 1:-1]  # up
    neighbors += padded_black[:, :, 1:-1, 2:]   # right
    neighbors += padded_black[:, :, 1:-1, :-2]  # left
    
    # A pixel is an edge if:
    # 1. It's not black itself (intensity > threshold)
    # 2. It has at least one black neighbor
    # 3. It's not completely surrounded by black pixels
    is_edge = ((1 - is_black) *  # Non-black pixels
               (neighbors > 0).float() *  # With at least one black neighbor
               (neighbors < 4).float())   # But not completely surrounded by black pixels
    
    # Expand is_edge to match the number of channels in mse_loss
    is_edge_expanded = is_edge.expand(-1, pred.size(1), -1, -1)
    
    # Create weight tensor with same shape as mse_loss
    weights = torch.ones_like(mse_loss)
    weights[is_edge_expanded.bool()] = edge_multiplier
    
    # Apply weights and return mean loss
    return (mse_loss * weights).mean()