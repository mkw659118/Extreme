from baselines import DLinear, P_sLSTM, PMDformer, NLinear, TimeMixer, WPMixer, iTransformer, Informer, FEDformer, FeTS, HMformer, PatchTST, TimesNet
from baselines.HMformer import HMformer
from exp.exp_base_net_baseline import BasicModel

class Model(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        if config.model == 'DLinear':
            self.model = DLinear.Model(config)
        elif config.model == 'PMDformer':
            self.model = PMDformer.Model(config)
        elif config.model == 'DLinear':
            self.model = DLinear.Model(config)
            
        elif config.model == 'NLinear':
            self.model = NLinear.Model(config)
        
        elif config.model == 'iTransformer':
            self.model = iTransformer.Model(config)
        elif config.model == 'informer':
            self.model = Informer.Model(config)
            
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
        
        elif config.model == 'lstm':
            self.model = LSTM.Model(config)

        else:
            raise ValueError(f'Unsupported model type: {config.model}')
        