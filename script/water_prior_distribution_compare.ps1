$ReservoirSensors = @(
    "reservoir_stor_4001_sof24",
    "reservoir_stor_4005_sof24",
    "reservoir_stor_4007_sof24",
    "reservoir_stor_4009_sof24",
    "reservoir_stor_4011_sof24"
)

$PredLens = @(8, 72)
$DModels = @(256)
$Configs = @("ExtremeLSTMMemoStudentTPriorConfig", "ExtremeLSTMMemoGaussianPriorConfig")

foreach ($cfg in $Configs) {
    foreach ($pred in $PredLens) {
        foreach ($sensor in $ReservoirSensors) {
            foreach ($dm in $DModels) {
                Write-Host ">> Running water prior compare | config=$cfg, pred_len=$pred, sensor=$sensor, d_model=$dm"
                python "run_train_last.py" `
                    --config "$cfg" `
                    --dataset "water" `
                    --reservoir_sensor "$sensor" `
                    --seq_len 360 `
                    --pred_len "$pred" `
                    --d_model "$dm" `
                    --c_in 3 `
                    --enc_in 3 `
                    --dec_in 3 `
                    --out_dim 3 `
                    --epochs 200 `
                    --patience 40 `
                    --train_volume 40000 `
                    --rounds 3 `
                    --oversampling 40 `
                    --loss_func "L1Loss" `
                    --bs 256 `
                    --seed 2026 `
                    --retrain True `
                    --num_experts 4 `
                    --top_k_experts 2 `
                    --retrieval_num 2 `
                    --pretrain_epochs 10 `
                    --state_prior_scales "1,4,8,16" `
                    --state_prior_include_seq_level True `
                    --logger "water_prior_distribution_compare"
            }
        }
    }
}
