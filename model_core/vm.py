import torch
from .ops import OPS_CONFIG
from .vocab import FORMULA_VOCAB

# ── 扩展后词表规模说明（task 12.1）──────────────────────────────────────────
#
# 本次扩展（factor-operator-library-expansion）后：
#   - 特征数 F  = len(FORMULA_VOCAB.feature_names)  （当前 65，覆盖 8 大类）
#   - 算子数 O  = len(OPS_CONFIG)                    （当前 66）
#   - 词表总 size = F + O = 131
#   - feat_offset = F = 65（feature token id ∈ [0, 64]）
#   - operator token id ∈ [F, F+O-1] = [65, 130]
#
# StackVM 的 op_map / arity_map **完全动态**从 FORMULA_VOCAB 与 OPS_CONFIG 派生，
# 不硬编码任何 token 数或偏移值，因此无需在此处做任何结构变更。后续再次扩展
# 特征或算子时只需更新注册表，VM 自动消费。
#
# Cross-sectional 算子（CS_RANK / CS_SCALE / CS_NEUTRALIZE）沿 N 维逐时间步
# 操作，输入/输出形状均为 [N, T]，满足统一的 [N,T]→[N,T] 契约（R2.9）；
# VM 主循环无需对它们做任何特殊处理。


class StackVM:
    def __init__(self):
        # feat_offset 动态从 FORMULA_VOCAB.operator_offset 读取（= feature_count = F）。
        # 扩展后 F=65，因此 feature token id ∈ [0, 64]，operator token id ∈ [65, 130]。
        self.feat_offset = FORMULA_VOCAB.operator_offset
        # op_map / arity_map 动态从 OPS_CONFIG 构建，算子总数 = len(OPS_CONFIG) = 66。
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}

    @staticmethod
    def _normalize_output(x: torch.Tensor) -> torch.Tensor:
        """
        对因子输出做标准化，确保幅度足够触发 neutral band 入场。

        策略（三级降级）：
        1. 截面 zscore（跨品种，每时间步）：适合因子跨品种有分散
        2. 时序 zscore（每品种，全局）：当截面 std 太小时使用
        3. 若两级都失败（因子是常数）：返回原值，由 const_cnt 拦截

        Returns:
            [N, T] clip 到 [-3, 3]，若是常数则返回原值（engine 会过滤）
        """
        N, T = x.shape

        # 检测是否是全局常数（标准化无意义）
        global_std = x.std()
        if global_std < 1e-6:
            return x   # 常数因子，由 engine 的 const_cnt 拦截

        # ── 截面标准化（跨品种，每时间步）────────────────────────────
        cs_mean = x.mean(dim=0, keepdim=True)
        cs_std  = x.std(dim=0, keepdim=True).clamp(min=1e-8)
        cs_z    = (x - cs_mean) / cs_std

        if cs_z.std() >= 0.3:
            return torch.clamp(cs_z, -3.0, 3.0)

        # ── 时序标准化（每品种独立）─────────────────────────────────
        ts_mean = x.mean(dim=1, keepdim=True)
        ts_std  = x.std(dim=1, keepdim=True).clamp(min=1e-8)
        ts_z    = (x - ts_mean) / ts_std

        if ts_z.std() >= 0.1:
            return torch.clamp(ts_z, -3.0, 3.0)

        # ── 两级均失败：因子无区分度，返回原值让 engine 过滤 ────────
        return x

    def execute(self, formula_tokens, feat_tensor):
        stack = []
        try:
            for token in formula_tokens:
                token = int(token)
                if token < self.feat_offset:
                    if token >= feat_tensor.shape[1]:
                        return None
                    stack.append(feat_tensor[:, token, :])
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity: return None
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()
                    func = self.op_map[token]
                    res = func(*args)
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(res)
                else:
                    return None
            if len(stack) == 1:
                result = stack[0]
                # 最终输出标准化：保证因子幅度足够，避免全程空仓
                return self._normalize_output(result)
            else:
                return None
        except Exception:
            return None
