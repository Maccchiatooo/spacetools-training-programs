#!/bin/bash
# =============================================================================
# GFlowRL on SpaceTools —— 唯一入口
#
# 用法:   bash train.sh [实验名]
# 例:     bash train.sh beta5_bigbatch
#
# 所有可调参数都在下面这一段。系统层面的东西（环境版本、代码改动、数据路径、
# checkpoint）已经固定，不需要动，见 docs/。
# =============================================================================

# ============================ 你要调的参数 ==================================

# --- GFlowRL 算法（论文 arXiv 2607.13394 Eq.4/6/7/8）---
export GFLOWRL_BETA=2.5          # β 逆温度。目标分布 π_ref·exp(β·r)/Z 的锐度。
                                 # 论文用 8（其 r 是 0/1 正确性）；本任务 reward 是
                                 # 0~10 复合分数，按 std(β·r) 匹配折算得 2.5。
                                 # 论文 Table 6a：β∈[1,10] 精度差异 <0.5 分。
export GFLOWRL_EPS_LOW=2.7       # 流量间隙裁剪下界 ε_low（Eq.7）
export GFLOWRL_EPS_HIGH=3.8      # 上界 ε_high，须 > ε_low（论文：给正向修正更多余量）
                                 # ⚠ 这两个值必须按实测 g 的分布定，见 docs/TUNING.md。
                                 # 判据：日志里 [GFLOWRL] raw_g 那行的 sat 应在 5~10%。

# --- 采样规模 ---
export FILTER_GROUPS=True         # 组内奖励全同的退化组置零（论文 Table 9）。
                                 # 那种组的 g 完全由 logprob 漂移决定，是纯噪声。
                                 # 实测退化率约 33%，有效 batch 会按此缩水，需调大 batch 补偿。

export ROLLOUT_N=16              # G，每个 prompt 的 rollout 数。Z 是组内蒙特卡洛估计，
                                 # G 太小则 Z 噪声大。论文用 16。
export TRAIN_BATCH_SIZE=63     # baseline 取官方 SpaceTools GRPO 的 64 附近值。
                                 # 约束：batch×G 须被 GPUS_PER_NODE 整除 → batch 须为 3 的倍数，
                                 # 故取 63（1008 轨迹/步，1008/6=168）；64 本身不满足。
                                 # 加 filter_groups 后有效 batch ≈ 42。
                                 # ⚠ 约束：TRAIN_BATCH_SIZE × ROLLOUT_N 必须能被
                                 # GPUS_PER_NODE 整除。21×16=336, 336/6=56 ✓

# --- 资源划分（单节点 8×B200）---
export TOOL_PG_GPUS=2            # 工具 actor 占的卡数。num_gpus 之和正好 2.000，零余量：
                                 # 改 actor 数或 fraction 前先跑 scripts/check_tool_packing.py
export GPUS_PER_NODE=6           # 训练用的卡数（FSDP + sglang rollout）
export GPU_MEM_UTIL=0.5          # sglang 显存占比。free_cache_engine=False 时它全程
                                 # 占着不放，需给训练留空间。

# --- 训练时长 ---
export TOTAL_EPOCHS=4            # 5500/63 = 87 步一轮；4 轮 ≈ 348 步，与论文 300 步同量级。
export SAVE_FREQ=10              # 每多少步存 checkpoint
export TEST_FREQ=20              # 验证集 350 条多轮工具调用，比一个训练步还重，调稀。

# ============================ 以下一般不用改 ================================

export FREE_CACHE_ENGINE=False   # torch_memory_saver 在本环境拒绝工作，见 docs/DEVIATIONS.md
export ATTN_BACKEND=flashinfer   # fa3 不支持 Blackwell
export MM_ATTN_BACKEND=triton_attn
export ATTN_IMPL=sdpa
export USE_REMOVE_PADDING=True

RELEASE=/mnt/shared/home/yuyu/0830/gflowrl_release
RUN_ID="${1:-run$(date +%m%d_%H%M)}"
export GFLOWRL_RUN_ID="$RUN_ID"

echo "=== GFlowRL: $RUN_ID ==="
echo "  β=$GFLOWRL_BETA  ε=($GFLOWRL_EPS_LOW,$GFLOWRL_EPS_HIGH)  G=$ROLLOUT_N  batch=$TRAIN_BATCH_SIZE"
echo "  GPU: 工具 $TOOL_PG_GPUS / 训练 $GPUS_PER_NODE"
echo "  日志: $RELEASE/logs/train_$RUN_ID.log"
echo

bash "$RELEASE/scripts/preflight.sh" || { echo "预检失败，已中止"; exit 1; }

source /mnt/shared/home/yuyu/opt/miniconda3/etc/profile.d/conda.sh
conda activate ray-client312
export RAY_ADDRESS=http://127.0.0.1:8265
ENVJSON=$(python - <<PY
import json,os
keys=["GFLOWRL_RUN_ID","GFLOWRL_BETA","GFLOWRL_EPS_LOW","GFLOWRL_EPS_HIGH","ROLLOUT_N",
      "TRAIN_BATCH_SIZE","TOOL_PG_GPUS","GPUS_PER_NODE","GPU_MEM_UTIL","TOTAL_EPOCHS",
      "SAVE_FREQ","TEST_FREQ","FILTER_GROUPS","FREE_CACHE_ENGINE","ATTN_BACKEND","MM_ATTN_BACKEND",
      "ATTN_IMPL","USE_REMOVE_PADDING"]
print(json.dumps({"env_vars":{k:os.environ[k] for k in keys if k in os.environ}}))
PY
)
ray job submit --no-wait --submission-id "gflowrl-$RUN_ID" \
  --runtime-env-json "$ENVJSON" -- \
  /mnt/shared/home/yuyu/opt/miniconda3/envs/ray-client312/bin/python \
  "$RELEASE/scripts/submit_gflowrl_private.py" 86400

echo
echo "跟踪:  tail -f $RELEASE/logs/train_$RUN_ID.log"
echo "状态:  ray job status gflowrl-$RUN_ID"
echo "停止:  ray job stop  gflowrl-$RUN_ID"
