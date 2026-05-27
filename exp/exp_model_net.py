from baselines import DLinear, PMDformer
from exp.exp_base_net_baseline import BasicModel

class Model(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        if config.model == 'DLinear':
            self.model = DLinear.Model(config)
        elif config.model == 'PMDformer':
            self.model = PMDformer.Model(config)
        
        else:
            raise ValueError(f'Unsupported model type: {config.model}')
