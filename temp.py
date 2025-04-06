if __name__ == '__main__':
    from modules.pretrain_timer import Timer
    import torch
    from utils.exp_config import get_config

    config = get_config()
    # config.patch_len = 96
    # config.d_model = 1024
    # config.d_ff = 2048
    # config.e_layers = 8
    # config.n_heads = 8
    # config.dropout = 0.10
    # config.factor = 1
    # config.output_attention = 1
    # config.activation = 'gelu'
    # config.ckpt_path = 'Timer_forecast_1.0.ckpt'
    backbone = Timer(config)
    ckpt_path = 'Timer_forecast_1.0.ckpt'
    sd = torch.load(ckpt_path, weights_only=False, map_location="cpu")["state_dict"]
    sd = {k[6:]: v for k, v in sd.items()}
    backbone.load_state_dict(sd, strict=True)
    print(backbone)