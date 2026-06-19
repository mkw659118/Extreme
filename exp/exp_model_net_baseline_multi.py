from baselines import PMDformer, WPMixer, iTransformer, FEDformer, FeTS, HMformer, PatchTST, TimesNet, P_sLSTM
from baselines.HMformer import HMformer
from baselines.xLSTMTime import xLSTMTime
from baselines.xlstm_mixer import xLSTMMixer
from exp.exp_base_net_baseline_multi import BasicModel

class Model(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        if config.model == 'PMDformer':
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
        
