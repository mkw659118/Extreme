# coding : utf-8
# Author : Yuxiang Zeng


# 在这里加上每个Batch的loss，如果有其他的loss，请在这里添加，
def compute_loss(model, pred, label):
    loss = model.loss_function(pred, label)
    try:
        # for i in range(len(model.model.encoder.layers)):
        #     loss += model.model.encoder.layers[i][3].aux_loss
        if model.config.dis_method == 'cosine':
            loss += 1e-3 * model.distance(pred, label)
        loss += 1e-3 * model.model.aux_loss
    except:
        pass

    return loss
