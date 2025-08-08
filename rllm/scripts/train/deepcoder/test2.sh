##!/bin/bash
set -x

# 环境变量设置 - 保持与之前成功的配置一致
# export MYPROMPT_TEMPLATE="my_prompt2" # 只有在 use_online_transform=True 时才会被使用
export TOKENIZERS_PARALLELISM=true

export WANDB_API_KEY=5fb87d5f0214b8686a9c1c14fa2b6411240ee277
export HF_TOKEN=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

export VALUE_BASELINE="optimal"

# ROCm相关环境变量 - 关键配置
export NCCL_DEBUG=INFO
export RCCL_MSCCLPP_ENABLE=1
export MSCCLPP_READ_ALLRED=1
export HIP_FORCE_DEV_KERNARG=1
export RCCL_MSCCLPP_THRESHOLD=1073741824
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1
export NCCL_MIN_NCHANNELS=112
export TORCH_BLAS_PREFER_HIPBLASLT=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
export FSDP_VERBOSE=1
export RAY_CGRAPH_get_timeout=6000
export NCCL_P2P_DISABLE=0
export NCCL_P2P=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_IB_HCA=mlx5_ib0,mlx5_ib1,mlx5_ib2,mlx5_ib3,mlx5_ib4,mlx5_ib5,mlx5_ib6,mlx5_ib7
export NCCL_IB_GID_INDEX=3
export RCCL_MSCCL_ENABLE=0
export GPU_MAX_HW_QUEUES=2
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# vLLM配置
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_USE_V1=1
export VLLM_USE_TRITON_FLASH_ATTN=1
export HIP_FORCE_DEV_KERNARG=0

# 显示推理进度
#export VLLM_LOGGING_LEVEL=WARNING
#export VERL_LOGGING_LEVEL=WARNING

# 设备可见性 - 只设置一个
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES  
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# wandb配置
#export WANDB_RESUME=never

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES" 
echo "ROCR_VISIBLE_DEVICES: $ROCR_VISIBLE_DEVICES"

WANDB_PROJECT=qwen3-code-grpo-deepcoder
EXP_NAME=qwen3-4b-base-8k-deepcoder-lcbv5-sync
MODEL_PATH=Qwen/Qwen3-4B-Base

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.mask_truncated_samples=True \
    data.train_files=/mnt/nvme5n1/markli/projects/qwen3_deepcoder/rllm/data/deepcoder_train.parquet \
    data.val_files=/mnt/nvme5n1/markli/projects/qwen3_deepcoder/rllm/data/test_livecodebench.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.shuffle=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=34816 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=34816 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    reward_model.enable=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=30 \
    trainer.validation_data_dir=/sgl-workspace/models/validation_results/${WANDB_PROJECT}/${EXP_NAME} \
    trainer.rollout_data_dir=/sgl-workspace/models/rollout_results/${WANDB_PROJECT}/${EXP_NAME} \
    trainer.default_local_dir=/sgl-workspace/models/${WANDB_PROJECT}/${EXP_NAME} "${@:1}" 