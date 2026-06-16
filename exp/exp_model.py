from baselines import DLinear, FeTS
from baselines import NLinear
from baselines import iTransformer
from baselines import Informer 
from baselines import FEDformer 
from baselines import PMDformer
from baselines import TimesNet
from baselines.CrossFormer.Crossformer import Crossformer
from baselines.mlp_test import MLPTest
from baselines.Linear import Linear
from baselines.Linear2 import Linear2
from baselines.Linear3 import Linear3
from baselines.Linear4 import Linear4
from baselines.Linear5 import Linear5
from baselines.HMformer import HMformer
from baselines.SeasonalTrendModel import SeasonalTrendModel
from baselines.DFTDecomModel import DFTDecomModel
from baselines.Transformer import Transformer
from modules.ExtremeLSTM import ExtremeLSTM
# from modules.ExtremeLSTMMemoMoENew import ExtremeLSTMMemo
# from modules.ExtremeLSTMMemoNew import ExtremeLSTMMemo
from modules.RMF666 import ExtremeLSTMMemo
from modules.MoEMemoFormer import ThreeExpertPatchTransformer
from baselines.MCANN.Group_GMM5 import DAN
from exp.exp_base import BasicModel
# from exp.exp_base_rmf666 import BasicModel
from baselines.encoder_seq import SeqEncoder

from modules.RMF_net import ExtremeLSTMMemo
from baselines import PMDformer, WPMixer, iTransformer, FEDformer, FeTS, HMformer, PatchTST, TimesNet, P_sLSTM
from baselines.HMformer import HMformer
from baselines.xLSTMTime import xLSTMTime
from baselines.xlstm_mixer import xLSTMMixer
from exp.exp_base import BasicModel

class Model(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_size = config.input_size
        self.hidden_size = config.rank

        if config.model == 'patch_extreme_memory_transformer': 
            self.model = ThreeExpertPatchTransformer(
                seq_len=config.seq_len,
                pred_len=config.pred_len,
                patch_len=config.patch_len,
                d_model=config.d_model,
                win_size=config.win_size,
                revin=config.revin,
                num_heads=config.n_heads,
                use_memory=config.use_memory,
                num_layers_intra_patch=config.num_layers_intra_patch,
                num_layers_inter_patch=config.num_layers_inter_patch,
                config=config
            )

        elif config.model == 'extreme_lstm_memo':
            self.model = ExtremeLSTMMemo(
                c_in=config.c_in,
                seq_len=config.seq_len,
                pred_len=config.pred_len,
                d_model=config.d_model,
                e_layers=config.e_layers,
                d_layers=config.d_layers,
                config=config
            )

        elif config.model == 'extreme_lstm':
            self.model = ExtremeLSTM(
                seq_len=config.seq_len,
                pred_len=config.pred_len,
                patch_len=config.patch_len,
                d_model=config.d_model,
                win_size=config.win_size,
                revin=config.revin,
                num_heads=config.n_heads,
                use_memory=config.use_memory,
                num_layers_intra_patch=config.num_layers_intra_patch,
                num_layers_inter_patch=config.num_layers_inter_patch,
                config=config
            )
        
        # 2025年9月8日21:34:04 测试水文数据集效果
        elif config.model == 'mlp_test':
            self.model = MLPTest(self.input_size, config)

        # 2025年05月30日11:45:49 这里只使用了一层的Linear，效果：
        elif config.model == 'mlp':
            self.model = Linear(self.input_size, config)

        # 2025年6月2日16:06:43 两层MLP,效果
        elif config.model == 'mlp2':
            self.model = Linear2(self.input_size, config)

        # 2025年6月2日16:08:17 三层MLP效果
        elif config.model == 'mlp3':
            self.model = Linear3(self.input_size, config)

        elif config.model == 'mlp4':
            self.model = Linear4(self.input_size, config)

        elif config.model == 'mlp5':
            self.model = Linear5(self.input_size, config)

        elif config.model == 'seasonal_trend_model':
            self.model = SeasonalTrendModel(config)

        elif config.model == 'dft':
            self.model = DFTDecomModel(config)

        elif config.model == 'transformer':  # 添加 transformer 支持
            self.model = Transformer(
                input_size=config.input_size,
                d_model=config.d_model,
                revin=config.revin,
                num_heads=config.n_heads,
                num_layers=config.num_layers,
                seq_len=config.seq_len,
                pred_len=config.pred_len,
                match_mode=config.match_mode,
                win_size=config.win_size,
                patch_len=config.patch_len,
                device=config.device
            )

        

        elif config.model in ['rnn', 'lstm', 'gru']:
            self.model = SeqEncoder(
                input_size=self.input_size,
                d_model=self.hidden_size,
                seq_len=config.seq_len,
                pred_len=config.pred_len,
                num_layers=config.num_layers,
                seq_method=config.model,
                bidirectional=True
            )
        elif config.model == 'crossformer':  # 添加 Crossformer 支持
            self.model = Crossformer(
                data_dim=self.input_size,
                in_len=config.seq_len,
                out_len=config.pred_len,
                seg_len=config.seg_len,  # 使用 config.seg_len
                win_size=4,
                d_model=self.hidden_size,
                n_heads=8,
                e_layers=2,
                dropout=0.1,
                device=config.device
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
            raise ValueError(f"Unsupported model type: {config.model}")


        