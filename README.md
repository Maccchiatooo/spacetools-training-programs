# SpaceTools 训练程序：SFT + GFlowRL

两阶段。SFT 在 Qwen2.5-VL-3B 上做工具调用的全参微调；GFlowRL 在它产出的
checkpoint 之上做强化学习，替换 SpaceTools 官方的 GRPO。

- **GFlowRL** = arXiv 2607.13394，*Scaling Distribution-Matching RL to Large
  Language Models*。**不是 FlowRL**（2509.15207），两者的配分函数处理方式不同。

## 五个文件

| 文件 | 行数 | 是什么 |
|---|---|---|
| `train_sft.sh` | 60 | SFT 入口：GPU 节点上的启动器 |
| `algo_sft_run_sft.sh` | 304 | SFT 流程本体（官方脚本，**逐字未改**） |
| `train_gflowrl.sh` | 90 | GFlowRL 入口：参数 + 预检 + 提交 |
| `algo_gflowrl_core_algos.py` | 2606 | GFlowRL 算法本体（完整文件，真实运行的那份） |
| `algo_gflowrl_excerpt.py` | 232 | ↑ 其中 GFlowRL 的那一段，便于阅读 |

**两边的"本体"性质不同，别混为一谈：**

SFT 侧**没有自研算法**。训练算法就是 LLaMA-Factory 的标准监督微调，
`algo_sft_run_sft.sh` 是官方 SpaceTools 的流程脚本，我们一行没改。
所有对本集群的适配都以环境变量的形式放在 `train_sft.sh` 里。

GFlowRL 侧才是写了算法的地方。论文官方代码未发布
（github.com/microsoft/gflowrl 为 404），按 Eq.4/6/7/8 自行实现。

## SFT

Qwen2.5-VL-3B 全参微调，8×B200 + DeepSpeed ZeRO-3，3000 步 / 3.08 轮。

```
train loss  0.7452 → 0.0532
eval  loss  0.6836 → 0.0482
```

权重验证过：语言侧确实被训练（抽查 6/6 与 base 不同），视觉侧保持冻结
（4/4 相同，符合 `freeze_vision_tower: true`）。

`algo_sft_run_sft.sh` 的三个阶段：
1. 下载数据，把 11 个工具的 schema 注入每条样本的 system prompt，
   v1 过滤掉含 robot 工具的样本（留 7020 条）
2. 生成 config，用 LLaMA-Factory 跑全参微调
3. 修 checkpoint 以兼容 RL：删 `config.json` 的 `text_config`、
   置 `tie_word_embeddings=True`、从 base model 拷 `preprocessor_config.json`

## GFlowRL

`algo_gflowrl_core_algos.py` 第 **2375~2606** 行是 GFlowRL（= `algo_gflowrl_excerpt.py`
的内容）。前面 2374 行是 verl 自带的 GAE / GRPO / RLOO / REINFORCE++ 等估计器。

通过 verl 的两个插件点接入，**不改 forward、不加参数**：

| 函数 | 装饰器 | 对应论文 |
|---|---|---|
| `compute_gflowrl_flow_gap()` | `@register_adv_est("gflowrl")` | Eq.4 组内蒙特卡洛估配分函数 Z、Eq.6 流量间隙 g、Eq.7 非对称裁剪 |
| `compute_policy_loss_gflowrl()` | `@register_policy_loss("gflowrl")` | Eq.8 平方残差损失 + 序列级重要性采样权重 |

与 FlowRL 的关键差别：配分函数不是待学习的网络（ProjZModule），而是**组内蒙特卡洛
均值**，所以不需要 `output_hidden_states`、不需要额外参数。

### 两处论文没写清、由实测定的参数

**β（逆温度）= 2.5，论文用 8。** 论文的 8 假设 0/1 奖励；本项目是 0~10 的复合奖励，
按 `std(β·r)` 对齐后得 2.5。

**ε = (2.7, 3.8)，不是 Table 9 的 0.2/0.28。** 那两个数是 DAPO 的 ratio clip，
不是 Eq.7 的流量间隙裁剪——用在这里会裁掉 88.7% 的样本。

按步实测：ε=2.7 落在 |g| 的 p99(≈2.2) 之外，所以它**放过分布主体、只夹住离群值**
（g_max<3 的步 sat=0.000；g_max 11.7/15.8/20.1 的步 sat=0.018/0.011/0.004）。
这是尾部保险该有的行为。

### 数学校验

`mean(g)=0.0000`（构造上应为 0）、`is_weight=1.0`（on-policy 首个 epoch），
Eq.4/6/7 手算一致。

## 运行状态（2026-09-03 08:22）

`gflowrl-baseline4` 运行中，第 **39** 步，3 个 checkpoint。
OOM / ActorDied / SIGSEGV 全为 0。步时约 13 分钟，348 步（4 轮）预计 ≈ 3 天。

39 步的趋势检验，**四项全不显著**——这个阶段还看不出在不在学：

| 指标 | 均值 | 斜率/步 | t |
|---|---|---|---|
| 奖励 `critic/score/mean` | 0.796 | −0.0004 | −0.25 |
| 退化组占比 | 0.533 | +0.0013 | +1.52 |
| entropy | 0.107 | −0.00001 | −0.08 |

「退化组」= 组内 16 条 rollout 奖励完全相同，被 `filter_groups` 置零、不产生梯度。
稳定在一半左右，意味着有效梯度来自约 30 个组/步。

这是第一个参数与文档一致的运行。前三次都作废了：工具挤爆 GPU0 的 OOM、
被 auto-resume 污染的续训、参数被中间层覆盖成 β=8.0/batch=21。

## 完整归档

这五个文件只是入口和算法。完整的训练程序（含 verl、11 个工具的 toolshed、
GraspGen、环境构建脚本、预检、全部日志）在：

```
0724/SFT/    SFT 归档，README.md 里有详细说明
0724/RL/     GFlowRL 归档，README.md + docs/DEVIATIONS.md + docs/TUNING.md
```

`DEVIATIONS.md` 逐条记录了与两篇论文的所有偏离及理由，包括哪些是硬件/集群
强制的、哪些是实测定的。
