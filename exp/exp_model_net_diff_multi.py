from modules.DARNet1 import ExtremeLSTMMemo
from modules.DATP_Net_multi import ExtremeLSTMMemo as DATPNet
from baselines import DLinear, PMDformer
from exp.exp_base_DARNet_multi import BasicModel


class Model(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.model == 'net':
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
        elif config.model in {'datp_net_multi', 'datp_net_step_multi', 'datp_net_horizon_multi'}:
            input_dim = getattr(config, 'enc_in', config.c_in)
            dec_in = getattr(config, 'dec_in', input_dim)
            out_dim = getattr(config, 'out_dim', dec_in)
            self.model = DATPNet(
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
        elif config.model == 'DLinear':
            self.model = DLinear.Model(config)
        elif config.model == 'PMDformer':
            self.model = PMDformer.Model(config)
        else:
            raise ValueError(f'Unsupported model type: {config.model}')
