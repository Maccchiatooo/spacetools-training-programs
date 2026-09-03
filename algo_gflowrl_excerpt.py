# =============================================================================
# GFlowRL: Scaling Distribution-Matching RL to Large Language Models
# arXiv 2607.13394. 官方代码未发布（github.com/microsoft/gflowrl 为 404），
# 以下按论文 Eq. 4/6/7/8 实现。
#
# 与 FlowRL 的关键差别：配分函数不再是一个待学习的网络（ProjZModule），而是
# 组内蒙特卡洛均值，因此不需要改 forward、不需要 output_hidden_states、
# 不需要额外参数。整套东西落在 verl 现成的两个插件点上：
#   - advantage estimator: 算逐序列的流量间隙 g̃，塞进 advantages 槽位
#   - policy loss:         算重要性权重 w 和平方残差
# =============================================================================


def _gflowrl_cfg(config, name: str, default):
    """从 AlgoConfig / DictConfig 里取 GFlowRL 超参，缺省时回退到论文默认值。"""
    if config is None:
        return default
    value = getattr(config, name, None)
    if value is None and hasattr(config, "get"):
        value = config.get(name, None)
    return default if value is None else value


def _group_mean(values: torch.Tensor, index: np.ndarray) -> torch.Tensor:
    """逐组求均值并广播回每个样本。index 是 uid 数组，同一 prompt 的 G 条 rollout 共享一个 uid。

    不用 verl.utils.group_mean_std：它对单元素组回退成 mean=0，而这里单元素组的
    正确行为是取该元素自身的值（Z 的定义就是组内均值，G=1 时均值即自身）。
    """
    gidx = as_torch_index(index).to(values.device)
    num_groups = int(gidx.max().item()) + 1
    sums = torch.zeros(num_groups, dtype=values.dtype, device=values.device)
    counts = torch.zeros(num_groups, dtype=values.dtype, device=values.device)
    sums.scatter_add_(0, gidx, values)
    counts.scatter_add_(0, gidx, torch.ones_like(values))
    return (sums / counts.clamp(min=1.0))[gidx]


@register_adv_est("gflowrl")
def compute_gflowrl_flow_gap(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    ref_log_prob: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GFlowRL 的"优势"其实是裁剪后的流量间隙 g̃（论文 Eq. 4/6/7）。

    之所以放在 advantage estimator 里算：这里是唯一同时拿得到 **组结构**（index）、
    **标量奖励**、**π_ref / π_old 的 logprob** 的地方；policy loss 拿不到前两者。
    g̃ 逐序列是常数，广播成 token 级后走 advantages 槽位传给 policy loss。

        Z_t(x) = (1/G) Σ_i [ β·r_i + logπ_ref(y_i|x) − logπ_old(y_i|x) ]        (Eq. 4)
        g_i    = sg[Z_t(x)] + (1/|y_i|)·log(π_old(y_i|x)/π_ref(y_i|x)) − β·r_i  (Eq. 6)
        g̃_i    = clip(g_i, −ε_low, +ε_high)                                     (Eq. 7)

    ⚠️ 论文内部不一致：Eq. 4 里的 logπ 未做长度归一化，Eq. 6 里做了。两者量纲差
    一个 |y|（数千 token）。训练起点 π_old == π_ref，该项恒为 0，两种读法一致；
    但随着策略漂移，未归一化的读法下这一项会以每 token 的漂移量 × 数千累积，
    量级远超 β·r（约 0..8），Z 会被漂移完全支配、失去 baseline 的作用。
    因此默认对两者都做长度归一化（gflowrl_normalize_z_logp=True）。
    设成 False 可得到 Eq. 4 的字面实现。
    """
    assert ref_log_prob is not None, (
        "GFlowRL 需要 ref_log_prob。请确保 actor_rollout_ref.actor.use_kl_loss=True "
        "（系数可为 0）以便 verl 启动 reference policy worker。"
    )
    assert old_log_probs is not None, "GFlowRL 需要 old_log_probs（rollout 策略的 logprob）。"

    beta = float(_gflowrl_cfg(config, "gflowrl_beta", 8.0))
    eps_low = float(_gflowrl_cfg(config, "gflowrl_eps_low", 0.2))
    eps_high = float(_gflowrl_cfg(config, "gflowrl_eps_high", 0.28))
    normalize_z = bool(_gflowrl_cfg(config, "gflowrl_normalize_z_logp", True))

    with torch.no_grad():
        response_mask = response_mask.to(old_log_probs.dtype)
        # response_mask 由 agent loop 构造：assistant 生成的 token 为 1，工具返回/user
        # 消息/padding 为 0（tool_agent_loop.py:289/431/463）。所以下面的求和与 |y|
        # 天然排除了工具返回的 token。
        seq_len = response_mask.sum(dim=-1).clamp(min=1.0)
        scores = token_level_rewards.sum(dim=-1)
        logp_old = (old_log_probs * response_mask).sum(dim=-1)
        logp_ref = (ref_log_prob * response_mask).sum(dim=-1)

        # log π_ref − log π_old，按需长度归一化（见 docstring 的不一致说明）
        ref_minus_old = logp_ref - logp_old
        z_logp_term = ref_minus_old / seq_len if normalize_z else ref_minus_old

        z = _group_mean(beta * scores + z_logp_term, index)          # Eq. 4
        flow_gap = z - ref_minus_old / seq_len - beta * scores       # Eq. 6
        clipped = flow_gap.clamp(min=-eps_low, max=eps_high)         # Eq. 7

        # 退化组过滤（论文 Table 9 的 "Filter groups: Accuracy-based"）。
        # 组内 G 条 rollout 奖励全同时，β·r_i 对组内每条都一样，于是
        #     g_i = Z − β·r_i − (1/|y_i|)log(π_ref/π_old)
        # 完全由 logprob 漂移项决定 —— 纯噪声，不含任何奖励信息。
        # 拿它当梯度信号只会注入噪声，所以整组置零。
        #
        # 与 DAPO 原版的差别：recipe/dapo/dapo_ray_trainer.py:266-284 是丢掉退化组后
        # **继续生成新 batch 直到凑够 train_batch_size**，有效 batch 不变、rollout 变多；
        # 这里只是置零，有效 batch 会按退化率缩水（实测约 33%）。
        # 需要靠调大 TRAIN_BATCH_SIZE 补偿。改主循环风险大，先用这个版本。
        if bool(_gflowrl_cfg(config, "gflowrl_filter_groups", True)):
            gidx = as_torch_index(index).to(scores.device)
            ng = int(gidx.max().item()) + 1
            gs = torch.zeros(ng, dtype=scores.dtype, device=scores.device)
            gq = torch.zeros(ng, dtype=scores.dtype, device=scores.device)
            gc = torch.zeros(ng, dtype=scores.dtype, device=scores.device)
            gs.scatter_add_(0, gidx, scores)
            gq.scatter_add_(0, gidx, scores * scores)
            gc.scatter_add_(0, gidx, torch.ones_like(scores))
            gvar = (gq / gc.clamp(min=1)) - (gs / gc.clamp(min=1)) ** 2
            keep = (gvar > 1e-8)[gidx].to(clipped.dtype)
            n_drop = int((1.0 - keep).sum().item())
            print(f"[GFLOWRL] filter_groups: 退化组置零 {int((gvar <= 1e-8).sum().item())}/{ng} 组, "
                  f"{n_drop}/{keep.numel()} 条轨迹；有效 batch 缩至 "
                  f"{100.0 * keep.mean().item():.1f}%", flush=True)
            clipped = clipped * keep

        # 裁剪前的 g 分布。论文（Sec.3.2 "Why this works"）明确裁剪应当是
        # "a safeguard on outlier rollouts ... inactive in a neighborhood of the
        # optimum"，即极少触发。若饱和率居高不下，说明 ε 与 β·r 的量纲不匹配
        # （Table 9 的 0.2/0.28 出自 GFlowRL/FlowRL/GRPO 共用表的 "Clip ratio" 行，
        # 更像 DAPO 的比值裁剪而非 Eq.7 的流量间隙裁剪）。打印分位数以便按实测定 ε。
        with torch.no_grad():
            q = torch.tensor([0.5, 0.9, 0.95, 0.99], device=flow_gap.device, dtype=torch.float32)
            fq = torch.quantile(flow_gap.float().abs(), q).tolist()
            sat = ((flow_gap <= -eps_low) | (flow_gap >= eps_high)).float().mean().item()
            # 退化组比例：GFlowRL 只从组内差异学习。若一组 G 条 rollout 的奖励全相同，
            # 则 Z = β·r̄ 使该组所有 g_i = 0，对梯度零贡献。论文 Table 9 有
            # "Filter groups: Accuracy-based" 专门丢掉这种组，而 verl 主 trainer
            # 没有实现（只在 recipe/dapo 里）。先测出比例，再决定要不要移植。
            gidx = as_torch_index(index).to(scores.device)
            ng = int(gidx.max().item()) + 1
            gsum = torch.zeros(ng, dtype=scores.dtype, device=scores.device)
            gsq = torch.zeros(ng, dtype=scores.dtype, device=scores.device)
            gcnt = torch.zeros(ng, dtype=scores.dtype, device=scores.device)
            gsum.scatter_add_(0, gidx, scores)
            gsq.scatter_add_(0, gidx, scores * scores)
            gcnt.scatter_add_(0, gidx, torch.ones_like(scores))
            gvar = (gsq / gcnt.clamp(min=1)) - (gsum / gcnt.clamp(min=1)) ** 2
            degenerate = (gvar <= 1e-8).float().mean().item()
            print(f"[GFLOWRL] groups: n={ng} 退化组(组内奖励全同)={degenerate:.3f} "
                  f"组内奖励std均值={gvar.clamp(min=0).sqrt().mean().item():.4f}", flush=True)

            print(f"[GFLOWRL] raw_g: mean={flow_gap.mean().item():.4f} std={flow_gap.std().item():.4f} "
                  f"|g|_p50={fq[0]:.4f} p90={fq[1]:.4f} p95={fq[2]:.4f} p99={fq[3]:.4f} "
                  f"max={flow_gap.abs().max().item():.4f} | eps=({eps_low},{eps_high}) sat={sat:.3f} "
                  f"| beta*r: mean={(beta*scores).mean().item():.3f} std={(beta*scores).std().item():.3f}",
                  flush=True)

        advantages = clipped.unsqueeze(-1) * response_mask

    return advantages, advantages


@register_policy_loss("gflowrl")
def compute_policy_loss_gflowrl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """GFlowRL 训练损失（论文 Eq. 8）：

        L = (1/G) Σ_i w_i · ( g̃_i + (1/|y_i|)·log(π_θ(y_i|x)/π_old(y_i|x)) )²
        w_i = min( π_θ(y_i|x)/π_old(y_i|x), 1+ε )

    `advantages` 里装的是 advantage estimator 算好的 g̃（逐序列常数，token 级广播）。

    ⚠️ w_i 的偏差修正：论文正文写的是未归一化的序列比 π_θ(y)/π_old(y)，但那是数千个
    token 的 logprob 之和取 exp，数值上必然要么爆到阈值要么塌到 0，没有可用梯度信号。
    Table 9 同时列了 "Ratio scaling: Geometric"，指向几何平均比 exp((1/|y|)Σ log ratio)
    ——即 GSPO 那套序列级 IS。默认用几何平均比；gflowrl_raw_seq_ratio=True 可切回字面实现。
    """
    assert config is not None
    is_threshold = float(_gflowrl_cfg(config.policy_loss, "gflowrl_is_threshold", 2.0))
    raw_seq_ratio = bool(_gflowrl_cfg(config.policy_loss, "gflowrl_raw_seq_ratio", False))

    response_mask = response_mask.to(log_prob.dtype)
    seq_len = response_mask.sum(dim=-1).clamp(min=1.0)

    # g̃ 逐序列是常数，从 token 级 advantages 还原回序列级
    flow_gap = (advantages * response_mask).sum(dim=-1) / seq_len

    # (1/|y|)·log(π_θ/π_old)，梯度从这里进
    log_ratio_sum = ((log_prob - old_log_prob) * response_mask).sum(dim=-1)
    log_ratio_norm = log_ratio_sum / seq_len

    with torch.no_grad():
        exponent = log_ratio_sum if raw_seq_ratio else log_ratio_norm
        is_weights = torch.exp(exponent.clamp(max=10.0)).clamp(max=is_threshold)
        if rollout_is_weights is not None:
            # rollout correction 的权重是 token 级的，先压成序列级再相乘
            is_weights = is_weights * (
                (rollout_is_weights * response_mask).sum(dim=-1) / seq_len
            )

    residual = flow_gap + log_ratio_norm
    seq_loss = is_weights * residual.pow(2)

    # 摊回 token 级，让 agg_loss 的 "seq-mean-token-sum" 做跨 DP rank 的正确归一化：
    # sum_t(mask * loss_mat) 恰好还原 seq_loss，再对序列取均值。
    loss_mat = (seq_loss / seq_len).unsqueeze(-1).expand_as(response_mask)
    pg_loss = agg_loss(
        loss_mat=loss_mat,
        loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-sum",
        **config.global_batch_info,
    )

    # 裁剪饱和率：g̃ 长期贴边说明 ε 的量纲与 β·r 不匹配（见 GFLOWRL_PLAN.md 第 2 节的
    # ε_low/ε_high 歧义），需要重标定而不是继续训。
    eps_low = float(_gflowrl_cfg(config.policy_loss, "gflowrl_eps_low", 0.2))
    eps_high = float(_gflowrl_cfg(config.policy_loss, "gflowrl_eps_high", 0.28))
    eps_hit = ((flow_gap <= -eps_low + 1e-6) | (flow_gap >= eps_high - 1e-6)).float().mean()
    pg_metrics = {
        "actor/gflowrl_flow_gap": flow_gap.mean().detach().item(),
        "actor/gflowrl_flow_gap_abs": flow_gap.abs().mean().detach().item(),
        "actor/gflowrl_clip_saturation": eps_hit.detach().item(),
        "actor/gflowrl_is_weight": is_weights.mean().detach().item(),
        "actor/gflowrl_residual_abs": residual.abs().mean().detach().item(),
        "actor/ppo_kl": verl_F.masked_mean(old_log_prob - log_prob, response_mask).detach().item(),
        "actor/pg_clipfrac": 0.0,
        "actor/pg_clipfrac_lower": 0.0,
    }
    return pg_loss, pg_metrics
