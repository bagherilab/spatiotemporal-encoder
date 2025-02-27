import torch
import torch.nn as nn
import torch.nn.functional as F


class LogErrorLoss(nn.Module):
    """
    Log Error Loss with Box Blur for spatial data comparison.

    This loss computes the average log difference between a predicted image and
    a target image after applying a box blur to both. It supports multi-channel images.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the box blur kernel. Default is 3.
    epsilon : float, optional
        Small constant added to avoid log(0). Default is 1e-8.
    """

    def __init__(self, kernel_size: int = 3, epsilon: float = 1e-8) -> None:
        super(LogErrorLoss, self).__init__()
        self.epsilon = epsilon
        self.kernel_size = kernel_size

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the log error between predicted and target images.

        Parameters
        ----------
        predicted : torch.Tensor
            Predicted image tensor of shape (N, C, H, W), where N is the batch size,
            C is the number of channels, and H and W are the height and width.
        target : torch.Tensor
            Target image tensor of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the average log error across all pixels and channels.
        """
        assert predicted.shape == target.shape, (
            "Predicted and target images must have the same shape"
        )

        predicted_blurred = self.box_blur(predicted)
        target_blurred = self.box_blur(target)

        log_predicted = torch.log(predicted_blurred + self.epsilon)
        log_target = torch.log(target_blurred + self.epsilon)
        log_error = torch.abs(log_predicted - log_target)

        return log_error.mean()

    def box_blur(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a box blur to the input tensor using average pooling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Blurred tensor of the same shape as the input.
        """
        return F.avg_pool2d(
            x, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2
        )
