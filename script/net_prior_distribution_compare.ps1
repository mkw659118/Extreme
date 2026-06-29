$PredLens = @(5, 10, 15, 20)
$DModels = @(256)
$Datasets = @(
    @{ Name = "Abilene"; File = "Abilene_single.csv" },
    @{ Name = "Geant"; File = "Geant_single.csv" },
    @{ Name = "Seattle"; File = "Seattle_single.csv" }
)
$Configs = @("NetStudentTPriorConfig", "NetGaussianPriorConfig")

foreach ($dataset in $Datasets) {
    foreach ($cfg in $Configs) {
        foreach ($pred in $PredLens) {
            foreach ($dm in $DModels) {
                Write-Host ">> Running prior distribution compare: config=$cfg, dataset=$($dataset.Name), pred_len=$pred, d_model=$dm"

                python "run_train_DARNet.py" `
                    --config "$cfg" `
                    --pred_len "$pred" `
                    --d_model "$dm" `
                    --dataset "$($dataset.Name)" `
                    --data_file "$($dataset.File)" `
                    --epochs 200 `
                    --patience 40 `
                    --seq_len 96 `
                    --rounds 5 `
                    --logger "net_prior_distribution_compare" `
                    --loss_func "L1Loss" `
                    --bs 32 `
                    --seed 2026 `
                    --retrain True `
                    --use_state_prior True `
                    --use_retrieval True `
                    --use_missing_aware_encoding True `
                    --num_experts 4 `
                    --top_k_experts 2 `
                    --retrieval_num 2 `
                    --state_prior_scales "1,4,8,16" `
                    --state_prior_include_seq_level True
            }
        }
    }
}
