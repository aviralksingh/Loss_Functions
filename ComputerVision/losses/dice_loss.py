import torch
import torch.nn as nn
from torch.nn.functional import softmax


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Forward pass for Dice Loss.

        Args:
            inputs (Tensor): Predictions from the model, expected to be of shape (batch_size, num_classes, height, width).
            targets (Tensor): Ground truth labels, expected to be of shape (batch_size, height, width).


        Description:
        Smooth: A small constant (smooth) is added to the numerator and denominator to avoid division by zero,
        especially when there is no overlap between predictions and ground truth.

        One-Hot Encoding: The ground truth labels are converted to one-hot encoding to match the dimensions of the predictions.
        This is essential for multi-class segmentation.

        Dice Coefficient: The Dice coefficient measures the overlap between the predicted mask and the ground truth mask.
        The loss is computed as 1 - Dice coefficient.

        Flattening: Both the predicted and target tensors are flattened to simplify the Dice coefficient calculation.

        Key Points:
        Dice loss is particularly useful in tasks where you care more about the overlap between the predicted segmentation and the ground truth,
        which is common in medical image segmentation, for example. it works for multi-class segmentation as well as binary segmentation by setting num_classes appropriately.

        Returns:
            Tensor: Dice loss value.
        """

        # Ensure targets are on the same device as inputs
        targets = targets.to(inputs.device)

        # Apply softmax to get probabilities if inputs are logits
        inputs = torch.softmax(inputs, dim=1)

        # Convert targets to one-hot encoding
        targets = nn.functional.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()

        # Flatten the tensors
        inputs_flat = inputs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        # Compute Dice coefficient
        intersection = (inputs_flat * targets_flat).sum()

        dice_coeff = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)

        # Dice loss is 1 - Dice coefficient
        loss = 1 - dice_coeff

        return loss


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn(4, 5, 256, 256).to(device)  # Example input with batch_size=4, num_classes=5, height=256, width=256
    targets = torch.randint(0, 5, (4, 256, 256)).to(device)  # Example ground truth labels

    criterion = DiceLoss()
    loss = criterion(inputs, targets)
    print("Dice Loss:", loss.item())



