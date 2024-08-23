import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def focal_loss_multiclass(predictions, labels, alpha=0.25, gamma=2.0, device='cuda'):
    """
    Args:
        predictions: Tensor of shape (N, C, H, W) - Predicted logits for each class at each pixel.
        labels: One Hot Tensor of shape (N, H, W) - Ground truth labels with class indices (0 to C-1).
        alpha: Weighting factor for each class (either a scalar or a tensor of shape (C,)).
        gamma: Focusing parameter to reduce the relative loss for well-classified examples.
        device: Device to perform computations on ('cuda' or 'cpu').

    Returns:
        loss: Computed focal loss for the input.
    """
    # Ensure tensors are on the correct device
    predictions = predictions.to(device)
    labels = labels.to(device)

    # Convert predictions to probabilities using softmax across the channel dimension (class dimension)
    # probs = F.sigmoid(predictions)  # Shape: (N, C, H, W)
    probs = F.softmax(predictions, dim=1)  # Shape: (N, C, H, W)

    # Convert labels to one-hot encoding to match the shape of predictions
    labels_one_hot = F.one_hot(labels, num_classes=predictions.shape[1])  # Shape: (N, H, W, C)
    labels_one_hot = labels_one_hot.permute(0, 3, 1, 2)  # Shape: (N, C, H, W)

    # Compute the cross-entropy loss
    ce_loss = F.cross_entropy(predictions, labels, reduction='none')  # Shape: (N, H, W)

    # Compute the focal loss
    pt = torch.sum(probs * labels_one_hot, dim=1)  # Shape: (N, H, W)

    # Ensure alpha is broadcast correctly
    if isinstance(alpha, (float, int)):
        alpha_factor = alpha
    else:
        alpha = alpha.to(device)
        alpha_factor = torch.sum(alpha[None, :, None, None] * labels_one_hot, dim=1)  # Shape: (N, H, W)

    focal_weight = alpha_factor * (1 - pt).pow(gamma)

    loss = focal_weight * ce_loss
    return loss.mean()


def focal_loss_binary(predictions, labels, alpha=0.25, gamma=2.0):
    """
    Args:
        predictions: Tensor of shape (512, 1024) - Predicted logits for each pixel.
        labels: Tensor of shape (512, 1024) - Ground truth labels (0 or 1).
        alpha: Weighting factor for the rare class.
        gamma: Focusing parameter to reduce the relative loss for well-classified examples.

    Returns:
        loss: Computed focal loss for the input.
    """
    # Convert predictions to probabilities using sigmoid
    probs = torch.sigmoid(predictions)

    # Binary Cross Entropy loss for each pixel
    bce_loss = F.binary_cross_entropy(probs, labels.float(), reduction='none')

    # Compute the focal loss
    pt = torch.where(labels == 1, probs, 1 - probs)  # pt is p_t
    focal_weight = (alpha * labels + (1 - alpha) * (1 - labels)) * (1 - pt).pow(gamma)

    loss = focal_weight * bce_loss
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss implementation.

        Args:
            alpha (float or list): Weighting factor for each class. Default is 1 (no weighting).
            gamma (float): Focusing parameter to down-weight easy examples and focus more on hard examples. Default is 2.
            reduction (str): Specifies the reduction to apply to the output. Options are 'none', 'mean', or 'sum'. Default is 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor(alpha)
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)

    def forward(self, inputs, targets):
        """
        Forward pass for Focal Loss.

        Args:
            inputs (Tensor): Logits or raw outputs from the model, expected to be of shape (batch_size, num_classes, ...).
            targets (Tensor): Ground truth labels, expected to be of shape (batch_size, ...).

        Returns:
            Tensor: Focal loss value.
        """
        # Convert class labels to one-hot encoding
        targets = F.one_hot(targets, num_classes=inputs.size(1)).float()
        targets = targets.permute(0, 3, 1, 2)  # Adjust dimensions if needed

        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)

        # Compute focal loss
        pt = (1 - probs) * targets + probs * (1 - targets)
        log_pt = torch.log(pt)
        loss = -self.alpha * ((1 - pt) ** self.gamma) * log_pt

        # Reduce loss based on reduction method
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    # Example usage
    N, C, H, W = 1, 5, 512, 1024  # Example dimensions (batch size N, number of classes C, height H, width W)
    predictions = torch.randn(N, C, H, W, requires_grad=True)  # Replace with your prediction tensor
    labels = torch.randint(0, C, (N, H, W))                    # Replace with your label tensor

    # If alpha is a scalar (applies equally to all classes)
    alpha = 0.25

    # If alpha is a vector (different weight for each class)
    # alpha = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.25])  # Adjust according to your number of classes

    # Calculate the focal loss
    loss = focal_loss_multiclass(predictions, labels, alpha=alpha)

    print("Focal Loss:", loss.item())

    # Example usage
    # Assuming you have predictions, labels as input tensors
    predictions = torch.randn((512, 1024), requires_grad=True)  # Replace with your prediction tensor
    labels = torch.randint(0, 2, (512, 1024)).float()           # Replace with your label tensor
    # Calculate the focal loss
    loss = focal_loss_binary(predictions, labels)

    print("Focal Loss:", loss.item())

    inputs = torch.randn(4, 5, 32, 32)  # Example input with batch_size=4, num_classes=5, height=32, width=32
    targets = torch.randint(0, 5, (4, 32, 32))  # Example targets

    criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
    loss = criterion(inputs, targets)
    print("Focal Loss:", loss.item())
