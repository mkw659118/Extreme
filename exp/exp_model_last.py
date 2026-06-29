from modules.RMF_net import ExtremeLSTMMemo
from modules.RMF_net_prior_compare import ExtremeLSTMMemo as ExtremeLSTMMemoPriorCompare
from baselines import PMDformer, WPMixer, iTransformer, FEDformer, FeTS, HMformer, PatchTST, TimesNet, P_sLSTM
from baselines.HMformer import HMformer
from baselines.xLSTMTime import xLSTMTime
from baselines.xlstm_mixer import xLSTMMixer
from exp.exp_base_last import BasicModel

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
        elif config.model == 'extreme_lstm_memo_prior_compare':
            input_dim = getattr(config, 'enc_in', config.c_in)
            dec_in = getattr(config, 'dec_in', input_dim)
            out_dim = getattr(config, 'out_dim', dec_in)
            self.model = ExtremeLSTMMemoPriorCompare(
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
        
        elif config.model == 'PMDformer':
            self.model = PMDformer.Model(config)
       
        elif config.model == 'iTransformer':
            self.model = iTransformer.Model(config)
        
        elif config.model == 'FEDformer':
            self.model = FEDformer.Model(config)
            
        elif config.model == 'PMDformer':
            self.model = PMDformer.Model(config)
            
        elif config.model == 'FeTS':
            self.model = FeTS.Model(config)
            
        elif config.model == 'HMformer':
            self.model = HMformer(config)
        elif config.model == 'PatchTST':
            self.model = PatchTST.Model(config)
            
        elif config.model == 'timesnet':
            self.model = TimesNet.Model(config)

        elif config.model == 'WPMixer':
            self.model = WPMixer.Model(config)

        elif config.model == 'P_sLSTM':
            self.model = P_sLSTM.Model(config)
        
        elif config.model == 'xLSTMTime':
            self.model = xLSTMTime(pred_len=self.config.pred_len, seq_len=self.config.seq_len, enc_in=self.config.enc_in)

        elif config.model == 'xlstm_mixer':
            self.model = xLSTMMixer(pred_len=self.config.pred_len, seq_len=self.config.seq_len, enc_in=self.config.enc_in)

        else:
            raise ValueError(f'Unsupported model type: {config.model}')
