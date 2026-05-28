from baselines import DLinear, PMDformer, NLinear, iTransformer, Informer, FEDformer, FeTS, HMformer
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
        
        else:
            raise ValueError(f'Unsupported model type: {config.model}')
