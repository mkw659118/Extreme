from modules.RMF_last import ExtremeLSTMMemo
# from exp.exp_base_last import BasicModel
from exp.exp_base_net import BasicModel

class Model(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        if config.model == 'extreme_lstm_memo':
            self.model = ExtremeLSTMMemo(
                c_in=config.c_in,
                seq_len=config.seq_len,
                pred_len=config.pred_len,
                d_model=config.d_model,
                e_layers=config.e_layers,
                d_layers=config.d_layers,
                config=config,
            )

        else:
            raise ValueError(f'Unsupported model type: {config.model}')
