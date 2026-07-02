"""
main.py — 多因子训练入口（每品种独立训练）

使用方法：
    python main.py              # 训练全部品种
    python main.py XAUUSDm      # 只训练指定品种（命令行参数）

流程：
    1. 连接 MT5，加载全部品种数据（做一次时间轴对齐）
    2. 对 Config.SYMBOLS 中每个品种：
       a. 构建 SingleSymbolDataManager（单品种视图）
       b. 初始化 AlphaEngine（target_symbol 参数化）
       c. 训练 ModelConfig.TRAIN_STEPS 步
       d. 保存结果到 strategies/best_{symbol}.json
    3. 汇总输出各品种最优公式和得分
"""

import sys
import pathlib

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from data_pipeline.single_symbol_manager import SingleSymbolDataManager
from model_core.engine import AlphaEngine
from model_core.config import ModelConfig


def main(target_symbols: list[str] | None = None):
    """
    Args:
        target_symbols: 要训练的品种列表。None 表示训练 Config.SYMBOLS 中全部品种。
    """
    print(f"{'='*60}")
    print(f"  AlphaGPT 多因子训练")
    print(f"  TRAIN_STEPS={ModelConfig.TRAIN_STEPS}  "
          f"MAX_FORMULA_LEN={ModelConfig.MAX_FORMULA_LEN}  "
          f"BATCH_SIZE={ModelConfig.BATCH_SIZE}")
    print(f"{'='*60}")

    with MT5DataFetcher() as fetcher:
        # 一次性加载全部品种数据（交集时间轴对齐）
        multi_mgr = MT5DataManager(fetcher)
        multi_mgr.load()

        symbols_to_train = target_symbols or multi_mgr.symbols
        print(f"  准备训练品种: {symbols_to_train}")
        print()

        results = {}

        for symbol in symbols_to_train:
            if symbol not in multi_mgr.symbols:
                print(f"  [跳过] {symbol} 不在已加载数据中")
                continue

            print(f"\n{'─'*60}")
            print(f"  开始训练: {symbol}")
            print(f"{'─'*60}")

            # 单品种视图
            single_mgr = SingleSymbolDataManager(multi_mgr, symbol)

            # 每个品种重新初始化模型（互相独立，不共享权重）
            engine = AlphaEngine(
                data_manager=single_mgr,
                target_symbol=symbol,
            )
            engine.train()

            results[symbol] = {
                "best_score": engine.best_score,
                "formula":    engine.best_formula,
                "readable":   engine._decode_formula(engine.best_formula),
            }

    # 汇总
    print(f"\n{'='*60}")
    print(f"  训练完成汇总")
    print(f"{'='*60}")
    for sym, r in results.items():
        print(f"  {sym:12s}  BestScore={r['best_score']:.4f}")
        print(f"             {r['readable']}")
        strategy_path = pathlib.Path("strategies") / f"best_{sym}.json"
        print(f"             → {strategy_path}")
    print()


if __name__ == "__main__":
    # 支持命令行指定品种：python main.py XAUUSDm USTECm
    cli_symbols = sys.argv[1:] if len(sys.argv) > 1 else None
    main(target_symbols=cli_symbols)
