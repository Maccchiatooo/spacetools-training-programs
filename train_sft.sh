#!/bin/bash
# Runs ON the GPU node (invoked by submit_sft.py through a Ray task holding all
# 8 B200s). Sets up the environment run_sft.sh assumes, then hands off to it.
#
# run_sft.sh itself is used VERBATIM — no edits to the official script. Every
# adaptation to this cluster is an environment variable set here:
#
#   LD_LIBRARY_PATH  must be unset. The NGC image points it at py3.12 torch
#                    libs, which breaks every conda torch on this node.
#   conda hook       run_sft.sh calls `conda activate` but does not source the
#                    hook itself.
#   OUTPUT_DIR       run_sft.sh would otherwise write to
#                    official_repo/experiments/sft_v1_<timestamp>/.
#   BASE_MODEL       local path, so Phase 3 copies preprocessor_config.json
#                    from disk instead of re-fetching it from HuggingFace.
#   NPROC_PER_NODE   run_sft.sh declares NUM_GPUS but never uses it; LLaMA-
#                    Factory's launcher reads NPROC_PER_NODE (launcher.py:66),
#                    falling back to the visible device count.
set -euo pipefail

ROOT=/mnt/shared/home/yuyu/0724/spacetools
REPO=$ROOT/official_repo

unset LD_LIBRARY_PATH
source /mnt/shared/home/yuyu/opt/miniconda3/etc/profile.d/conda.sh

export VERSION=v1
export TOOL_CONFIG=$REPO/SpaceTools-RL/examples/toolshed/toolshed_v1_config.yaml
export BASE_MODEL=$ROOT/sft/base_model/Qwen2.5-VL-3B-Instruct
export OUTPUT_DIR=$ROOT/sft/output/sft_v1
export MAX_STEPS=3000
export NPROC_PER_NODE=8
export NUM_GPUS=8

# DeepSpeed warns that its Triton autotune cache defaults to NFS, which can hang
# the process on exit. /mnt/shared is NFS, so point both caches at node-local disk.
export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_$$
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Reuse the dataset already pulled to sft/data instead of re-downloading onto
# the node. run_sft.sh's snapshot_download(local_dir=$OUTPUT_DIR/sft_data/hf_download)
# sees a complete tree and only verifies it; the copytree that follows leaves
# our pristine copy untouched and rewrites only the copy under sft_data/.
mkdir -p "$OUTPUT_DIR/sft_data"
if [ ! -e "$OUTPUT_DIR/sft_data/hf_download" ]; then
    ln -s "$ROOT/sft/data/spacetools-sft" "$OUTPUT_DIR/sft_data/hf_download"
fi

echo "=== launch_sft.sh ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "visible: ${CUDA_VISIBLE_DEVICES:-all}"
echo

cd "$REPO/SpaceTools-SFT"
# `source`, not `bash`. run_sft.sh calls `conda activate`, which is a shell
# FUNCTION defined by the hook sourced above — functions are not inherited by a
# child bash, so running it as a subprocess fails with "Run 'conda init' before
# 'conda activate'". Sourcing keeps it in this shell, where conda is defined.
source scripts/spacetools/run_sft.sh
