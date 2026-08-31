"""DATP-Net-only model adapter.

Unlike the legacy shared adapter, this module never imports DARNet1. Removing
the old DARNet model implementation therefore does not affect DATP-Net startup.
"""

from exp.exp_base_DARNet import BasicModel
from modules.DATP_Net import ExtremeLSTMMemo as DATPNet


class Model(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.model not in {
            "datp_net",
            "datp_net_step",
            "datp_net_horizon",
        }:
            raise ValueError(
                "exp_model_DATPNet only supports DATP-Net variants, "
                f"got model={config.model!r}."
            )

        input_dim = getattr(config, "enc_in", config.c_in)
        dec_in = getattr(config, "dec_in", input_dim)
        out_dim = getattr(config, "out_dim", dec_in)
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
