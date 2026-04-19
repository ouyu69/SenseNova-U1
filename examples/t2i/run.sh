#!/usr/bin/env bash
# Submit a T2I inference job to the internal SCO ACP training platform. The
# payload simply invokes `examples/t2i/inference.py`; feel free to copy this
# file as a starting point for your own environment.
set -euo pipefail

model_path=${MODEL_PATH:-hf_models/hf_step10000_ema}
repo_root=/mnt/aigc/yanglei/program/tmp/github/SenseNova-U1
echo "Running t2i inference with model path: ${model_path}"

sco acp jobs create \
    --workspace-name=aigc \
    --aec2-name=umm \
    --job-name=u1_t2i_example \
    --priority=NORMAL \
    --container-image-url='registry.ms-sc-01.maoshanwangtech.com/ccr_2/cogvideox:20241128-19h44m09s' \
    --storage-mount='c83d08bc-2965-11ef-b8c5-929f74fd8884:/mnt/aigc/' \
    --training-framework=pytorch \
    --worker-nodes=1 \
    --worker-spec="N6lS.Iu.I80.1" \
    --command="whoami && echo \$MASTER_ADDR && echo \$MASTER_PORT && echo \$RANK && echo \$WORLD_SIZE && \
            export HF_HOME=/mnt/aigc/shared_data/cache/huggingface && echo \$HF_HOME && \
            export CUDA_LAUNCH_BLOCKING=1 && \
            cd ${repo_root} && \
            ${repo_root}/.venv/bin/python examples/t2i/inference.py \
            --model_path ${model_path} \
            --prompt \"A chubby cat made of 3D point clouds, translucent with a soft glow.\" \
            --output out.png"
