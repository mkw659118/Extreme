import torch.nn.functional as F

# # 在这里加上每个Batch的loss，如果有其他的loss，请在这里添加，
# def compute_loss(model, inputs, pred, label, config):
#     loss = model.loss_function(pred[:,:,0:1], label)
#     return loss

def compute_loss(model, inputs, pred, label, config):
    if pred.dim() == 3:
        pred = pred[:, :, 0]
    if label.dim() == 3:
        label = label[:, :, 0]
    loss = model.loss_function(pred, label)
    return loss