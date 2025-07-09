# coding : utf-8
# Author : Yuxiang Zeng


# 在这里加上每个Batch的loss，如果有其他的loss，请在这里添加，
def compute_loss(model, pred, label, config):
    loss = model.loss_function(pred, label)
    
    try:
        if config.diffusion:
            loss = loss * config.lamda + (1 - config.lamda) * model.model.diffusion_loss
        # for i in range(len(model.model.encoder.layers)):
        #     loss += model.model.encoder.layers[i][3].aux_loss
        # if model.config.dis_method == 'cosine':
        #     loss += 1e-3 * model.distance(pred, label)
        # loss += 1e-3 * model.model.aux_loss
    except Exception as e:
        print(f"Error in computing loss: {e}")
        pass

    return loss
