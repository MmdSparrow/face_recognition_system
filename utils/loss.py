import torch

from torch.nn import Module


class TripletLoss(Module):
    def __init__(self, alpha=0.2):
        super(TripletLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred):
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        def L2_normSq(x):
            return torch.sum(x**2, dim=-1)
        pos_dist = L2_normSq(positive - anchor)
        neg_dist = L2_normSq(negative - anchor)
        basic_loss = pos_dist - neg_dist + self.alpha
        loss_per_example = torch.maximum(basic_loss, torch.tensor(0.0, device=basic_loss.device))
        loss = torch.sum(loss_per_example)
        return loss