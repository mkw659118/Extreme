from modules.RMF_net import ExtremeLSTMMemo
# from exp.exp_base_last import BasicModel
from exp.exp_base_mcann_mul import BasicModel
from baselines.MCANN.Group_GMM5 import DAN

class Model(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        if config.model == 'extreme_lstm_memo':
            input_dim = getattr(config, 'enc_in', config.c_in)
            dec_in = getattr(config, 'dec_in', input_dim)
            out_dim = getattr(config, 'out_dim', dec_in)
            self.model = ExtremeLSTMMemo(
                c_in=input_dim,
                seq_len=config.seq_len,
                pred_len=config.pred_len,
                d_model=config.d_model,
                e_layers=config.e_layers,
                d_layers=config.d_layers,
                dec_in=dec_in,
                out_dim=out_dim,
                config=config,
            )
        
        elif config.model == 'mcann':  # 添加 transformer 支持
            self.model = DAN(config)

        else:
            raise ValueError(f'Unsupported model type: {config.model}')
