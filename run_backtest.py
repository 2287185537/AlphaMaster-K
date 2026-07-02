"""
run_backtest.py — 多因子组合回测脚本

用法：
    python run_backtest.py              # 加载 strategies/best_{symbol}.json（多因子模式）
    python run_backtest.py --single     # 加载 best_mt5_strategy.json（单公式兼容模式）

多因子模式：
    - 为每个品种加载各自的最优公式
    - 每个品种使用自己的因子独立生成信号
    - 汇总组合级统计
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from backtest_viz import BacktestEngine, BacktestChart, BacktestReport
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from model_core.vm import StackVM
from model_core.features import MT5FeatureEngineer
import torch


def decode_formula(tokens: list[int]) -> str:
    names = FORMULA_VOCAB.token_names
    return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}" for t in tokens)


def load_strategy(path: Path) -> dict | None:
    """加载策略文件，返回含 formula/vocab_version 的 dict，或 None。"""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"formula": data, "vocab_version": "legacy", "symbol": None}
    return data


def main():
    OUTPUT_DIR = "backtest_output"
    single_mode = "--single" in sys.argv

    # ── 1. 加载各品种策略 ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    if single_mode:
        # 兼容旧的单公式模式
        data = load_strategy(Path(Config.STRATEGY_FILE))
        if data is None:
            print(f"[ERROR] 找不到: {Config.STRATEGY_FILE}")
            sys.exit(1)
        formula = data["formula"]
        symbol_formulas = {sym: formula for sym in Config.SYMBOLS}
        print(f"  模式: 单公式（所有品种共用）")
        print(f"  公式: {decode_formula(formula)}")
    else:
        # 多因子模式：每品种独立公式
        symbol_formulas = {}
        missing = []
        for sym in Config.SYMBOLS:
            path = Path("strategies") / f"best_{sym}.json"
            data = load_strategy(path)
            if data is None:
                missing.append(sym)
                continue
            # vocab_version 校验
            ver = data.get("vocab_version", "unknown")
            if ver != VOCAB_VERSION:
                print(f"  [警告] {sym}: vocab_version={ver} 与当前 {VOCAB_VERSION} 不符，跳过")
                continue
            symbol_formulas[sym] = data["formula"]
            print(f"  {sym}: {decode_formula(data['formula'])}  (score={data.get('best_score', 'N/A')})")

        if missing:
            print(f"  [缺失策略] {missing}，这些品种将跳过")
        if not symbol_formulas:
            print("[ERROR] 没有找到任何有效策略文件，请先运行 main.py 训练")
            sys.exit(1)

    print(f"{'='*60}\n")

    # ── 2. 加载数据 ───────────────────────────────────────────────────
    print("正在连接 MT5 并拉取数据...")
    with MT5DataFetcher() as fetcher:
        mgr = MT5DataManager(fetcher)
        mgr.load()
        raw_dict  = mgr.raw_dict
        syms      = mgr.symbols
        T         = raw_dict["open"].shape[1]
        print(f"  品种: {syms}  T={T} bars\n")

        # ── 3. 为每个品种计算因子并跑回测 ──────────────────────────
        print("运行回测引擎（多因子模式）...")
        vm   = StackVM()
        feat = MT5FeatureEngineer.compute_features(raw_dict)  # [N, F, T]

        results = []
        for i, sym in enumerate(syms):
            if sym not in symbol_formulas:
                print(f"  [跳过] {sym}（无策略文件）")
                continue

            formula = symbol_formulas[sym]
            # 单品种因子计算（只取第 i 行）
            feat_i   = feat[i:i+1]          # [1, F, T]
            raw_i    = {k: v[i:i+1] for k, v in raw_dict.items()}  # [1, T]

            engine = BacktestEngine(formula=formula)
            sym_results = engine.run(raw_i, feat_i, [sym])
            results.extend(sym_results)

    # ── 4. 打印统计 ────────────────────────────────────────────────────
    # 为每个结果补充公式信息，用于报告
    for r in results:
        sym = r.symbol
        formula = symbol_formulas.get(sym, [])
        readable = decode_formula(formula)
        print(f"\n{'─'*50}")
        print(f"  {sym}: {readable}")
        print(f"{'─'*50}")
        print(f"  Sortino={r.sortino:+.4f}  PnL={r.total_return:+.4f}"
              f"  Trades={r.n_trades}  WinRate={r.win_rate:.1%}"
              f"  AvgHold={r.avg_hold_bars:.1f}bars  MaxDD={r.max_drawdown:.4f}")

    # ── 5. 组合级汇总 ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  组合级汇总")
    print(f"{'='*60}")
    total_pnl   = sum(r.total_return for r in results)
    n_positive  = sum(1 for r in results if r.total_return > 0)
    n_sortino_p = sum(1 for r in results if r.sortino > 0)
    avg_hold    = sum(r.avg_hold_bars for r in results) / len(results) if results else 0
    print(f"  全品种总 PnL       : {total_pnl:+.4f}")
    print(f"  正收益品种数       : {n_positive}/{len(results)}")
    print(f"  Sortino>0 品种数   : {n_sortino_p}/{len(results)}")
    print(f"  全品种平均持仓     : {avg_hold:.1f} bars")
    if results:
        best  = max(results, key=lambda r: r.sortino)
        worst = min(results, key=lambda r: r.sortino)
        print(f"  最佳品种           : {best.symbol} (Sortino={best.sortino:.2f})")
        print(f"  最差品种           : {worst.symbol} (Sortino={worst.sortino:.2f})")
        consistency_ok = n_sortino_p >= len(results) * 0.6
        print(f"  品种一致性(>=60%)  : {'PASS' if consistency_ok else 'FAIL'}")
    print(f"{'='*60}")

    # ── 6. 保存 JSON 报告 ─────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    report_data = {
        "mode": "single_formula" if single_mode else "multi_factor",
        "symbols": {},
    }
    for r in results:
        formula = symbol_formulas.get(r.symbol, [])
        report_data["symbols"][r.symbol] = {
            "formula": formula,
            "readable": decode_formula(formula),
            "sortino": round(r.sortino, 4),
            "total_return": round(r.total_return, 6),
            "max_drawdown": round(r.max_drawdown, 6),
            "n_trades": r.n_trades,
            "win_rate": round(r.win_rate, 4),
            "avg_hold_bars": round(r.avg_hold_bars, 2),
        }
    report_path = f"{OUTPUT_DIR}/multi_factor_report.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON 报告已保存 → {report_path}")

    # ── 7. 生成图表 ────────────────────────────────────────────────────
    print("\n生成图表...")
    chart = BacktestChart(max_bars=120)
    chart.plot_all(results, output_dir=OUTPUT_DIR)
    for r in results:
        saved = chart.plot_all_trade_zooms(r, output_dir=OUTPUT_DIR,
                                           pre_bars=25, post_bars=12, max_trades=10)
        print(f"  {r.symbol}: {len(saved)} 张缩放图")

    print(f"\n全部图表已保存至 {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
