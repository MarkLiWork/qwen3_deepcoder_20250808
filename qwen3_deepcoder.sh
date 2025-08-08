#!/bin/bash
CONTAINER_NAME="qwen3-deepcoder"

# 检查容器是否已存在
if docker ps -a | grep -q $CONTAINER_NAME; then
    echo "容器 $CONTAINER_NAME 已存在"
    if docker ps | grep -q $CONTAINER_NAME; then
        echo "容器正在运行，直接进入..."
        docker exec -it $CONTAINER_NAME /bin/bash
    else
        echo "启动现有容器..."
        docker start $CONTAINER_NAME
        docker exec -it $CONTAINER_NAME /bin/bash
    fi
else
    echo "创建新的持久化容器（拉满资源配置）..."
    docker run -it \
      --name $CONTAINER_NAME \
      --hostname qwen3-deepcoder \
      --restart unless-stopped \
      --privileged \
      --ipc=host \
      --pid=host \
      --network=host \
      --shm-size=1t \
      --ulimit memlock=-1:-1 \
      --ulimit stack=67108864 \
      --memory-swappiness=0 \
      --oom-kill-disable \
      --cpus="$(nproc)" \
      --cpuset-cpus="0-$((`nproc`-1))" \
      --security-opt seccomp=unconfined \
      --security-opt apparmor=unconfined \
      --cap-add=ALL \
      -v /mnt/nvme5n1/markli:/mnt/nvme5n1/markli \
      -v /sys:/sys \
      -v /proc:/proc \
      -v /dev:/dev \
      --device=/dev/kfd --device=/dev/dri \
      --workdir=/mnt/nvme5n1/markli \
      --env NVIDIA_VISIBLE_DEVICES=all \
      --env HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      --env OMP_NUM_THREADS=$(nproc) \
      --env MALLOC_ARENA_MAX=4 \
      --env PYTHONUNBUFFERED=1 \
      workspacegenaiacr.azurecr.io/zewenchi/rocm6.3.4:vllm-0.8.5-numa-patch-ubuntu-22.04
fi
