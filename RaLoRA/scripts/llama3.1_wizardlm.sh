#! /bin/bash
base_options="--train-dataset-name wizardlm_52k \
--eval-dataset-name wizardlm \
--model-name llama3 \
--tokenizer-name llama3 \
--output-path your_output_ckpt_path \
--tokenizer-path your_tokenizer_path \
--ckpt-path your_pretrained_ckpt_path \
--prompt-path your_prompt_path \
--tb-log-dir your_tb_log_dir \
--dataset-class-name iterable \
"

train_options="--tensorboard \
    --show-loss-step 1 \
    --epochs 1 \
    --mode sft \
    --batch-size-per-gpu 4 \
    --eval-batch-size-per-gpu 1 \
    --eval-interval 50 \
    --bf16 \
    --from-pretrained \
    --show-avg-loss-step 1 \
    --variant 8b \
    --save-interval 10000 \
    --gradient-accumulation-steps 8 \
    --device cuda \
    --max-len 1024 \
    --max-src-len 1024 \
    --eval-max-len 1024 \
    --eval-max-src-len 1024 \
    --zero-stage 2 \
    --lr 5e-5 \
    --warmup 0.03 \
    --auto-warmup-steps 10 \
    --auto-warmup-rate 0.05 \
    --weight-decay 5e-4 \
    --lr-decay-style cosine \
    --lr-decay-ratio 0.1 \
    --atten-type flash \
    --save-trainable \
    --diy-optimizer \
    "

lora_options="--use-lora \
    --lora-rank 8 \
    --lora-scaler 16 \
    --replace-modules wq_wk_wv_wo \
    --weight-a-init-method kaiming \
    --lora-rank 8 \
    --use-Ralora \
    --Ralora-n-steps 64 \
    --Ralora-dynamic-scaling \
    --Ralora-importance-type union_mean \
    --Ralora-max-rank 16 \
    --Ralora-min-rank 4 \
    --erank-max-power 5 \
    "

options="--experiment-name chat_test_${SEED} \
    $base_options \
    $train_options \
    $lora_options \
    --seed ${SEED} \
    "

run_cmd="deepspeed --include localhost:0,1 --master_port 16667 ../u_train.py $options"
echo ${run_cmd}
eval ${run_cmd}