"""平掉 MT5 账户上本策略 magic 的全部持仓。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from execution.trader import MT5Trader

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MT5 not available")
    sys.exit(1)


def main():
    trader = MT5Trader()
    trader.connect()
    magic = Config.MAGIC_NUMBER
    # 平掉账户全部持仓（含非本策略 magic）
    all_pos = list(trader.get_positions())
    if not all_pos:
        print("无持仓")
        trader.close()
        return

    symbols = sorted({p.symbol for p in all_pos})
    print(f"待平仓品种: {symbols}  共 {len(all_pos)} 笔")
    for sym in symbols:
        ok = trader.close_all_positions(sym, filter_magic=False)
        print(f"  {sym}: {'OK' if ok else 'PARTIAL/FAIL'}")

    remaining = trader.get_positions()
    print(f"剩余持仓: {len(remaining)}")
    trader.close()


if __name__ == "__main__":
    main()
