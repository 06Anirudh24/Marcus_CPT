#!/bin/bash
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate jpy

# setting up default hyperparameters
subsample_ratio=1.0 # change this parameter to run the scaling plot
model_name="allenai/Olmo-3-1025-7B"
# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr) lr="$2"; shift 2 ;;
        --rr) rr="$2"; shift 2;;
        --epochs) epochs="$2"; shift 2 ;;
        --bs) bs="$2"; shift 2 ;;
        --wd) wd="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --task_name) task_name="$2"; shift 2 ;;
        --subsample_ratio) subsample_ratio="$2"; shift 2 ;;
        --model_name) model_name="$2"; shift 2 ;;
        --run_eval) run_eval=true; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done
# gpu_count="${SLURM_GPUS_ON_NODE:-1}"
pretty_name=${model_name##*/}
pretty_name=$(echo "$pretty_name" | sed 's/-//g')
grad_acc=$((bs / 2))
# Then, use an if-else statement to set the run_name
if [ "$subsample_ratio" = "1.0" ]; then
    run_name="Olmo_4e5_raw-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-${pretty_name}"
else
    run_name="scaling-subsample_ratio${subsample_ratio}-${task_name}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-${pretty_name}"
fi
echo "Running experiment with run name: $run_name"
output_dir="/share/goyal/aa2886/Olmo_4e5_raw"

# To run multiple sbatches 
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID:-0} % 20000) ))
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# Execute the training command with the specific hyperparameters
# gradient_accumulation_steps: 2 for 2 GPUs, 4 for 1 GPU

python -m torch.distributed.run --nproc_per_node=2 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" /home/aa2886/Marcus_CPT/train.py \
    --model_name=$model_name \
    --block_size=2048 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=3 \
    --gradient_accumulation_steps=2 \
    --num_train_epochs=$epochs \
    --learning_rate=$lr \
    --rehersal_rate=$rr \
    --subsample_ratio=$subsample_ratio \
    --overwrite_output_dir=True \
    --task_name=$task_name \
    --logging_steps=1 \
    --run_name=$run_name \
    --bf16=True \
    --output_dir=$output_dir \
    --weight_decay=$wd \
    --warmup_ratio=$warmup \
    --eval_strategy="no" \
    --save_strategy="steps" \
    --save_steps=611 \
    --save_total_limit=10 \
    --lr_scheduler_type="cosine" \
    --log_level="info" \
    --fsdp="hybrid_shard auto_wrap" \
    --fsdp_config="/home/aa2886/Marcus_CPT/scripts/config/olmo_fsdp_config.json"
