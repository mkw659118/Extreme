"""Independent model registry for linear-imputation two-stage baselines."""

from importlib import import_module

from exp.exp_base_net_baseline import BasicModel


class Model(BasicModel):
    """Forecasting baselines used after linear interpolation."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        module_names = {
            "PMDformer": "baselines.PMDformer",
            "iTransformer": "baselines.iTransformer",
            "FEDformer": "baselines.FEDformer",
            "FeTS": "baselines.FeTS",
            "HMformer": "baselines.HMformer",
            "PatchTST": "baselines.PatchTST",
            "timesnet": "baselines.TimesNet",
            "WPMixer": "baselines.WPMixer",
            "P_sLSTM": "baselines.P_sLSTM",
            "xLSTMTime": "baselines.xLSTMTime",
            "xlstm_mixer": "baselines.xlstm_mixer",
        }

        if config.model not in module_names:
            raise ValueError(
                f"Unsupported linear-imputation baseline: {config.model}. "
                f"Available: {', '.join(module_names)}"
            )

        # Import only the requested baseline.  Some optional models have extra
        # dependencies, and they should not prevent unrelated baselines from
        # running in a minimal environment.
        module = import_module(module_names[config.model])
        if config.model == "HMformer":
            self.model = module.HMformer(config)
        elif config.model == "xLSTMTime":
            self.model = module.xLSTMTime(
                pred_len=config.pred_len,
                seq_len=config.seq_len,
                enc_in=config.enc_in,
            )
        elif config.model == "xlstm_mixer":
            self.model = module.xLSTMMixer(
                pred_len=config.pred_len,
                seq_len=config.seq_len,
                enc_in=config.enc_in,
                slstm_backend=getattr(config, "slstm_backend", "vanilla"),
            )
        else:
            self.model = module.Model(config)
