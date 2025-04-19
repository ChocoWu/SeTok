from torch import nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.weight = weight
    
    def forward(self, pred, target, loss_mask=None, loss_weight=None):
        mse_loss = self.mse(pred, target) 
        if loss_mask is not None:
            mse_loss = (mse_loss * loss_mask)
            weight = loss_mask.sum([-2, -1])
            weight += 1
            mse_loss = mse_loss.sum([-2, -1])
            mse_loss = (mse_loss / weight)
        else:
            mse_loss = mse_loss.mean([-3, -2, -1])
        return mse_loss.mean() * self.weight