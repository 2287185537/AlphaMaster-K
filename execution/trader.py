"""
execution/trader.py — MT5 订单执行模块

MT5Trader 负责通过 MetaTrader5 Python API 执行市价单买卖及账户查询。
全同步接口，无 asyncio。
"""
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    # 测试环境占位，使与真实 MT5 常量一致的整数值
    class _MT5Stub:
        ORDER_TYPE_BUY  = 0
        ORDER_TYPE_SELL = 1
        TRADE_ACTION_DEAL = 1
        TRADE_RETCODE_DONE = 10009

        def initialize(self):  # noqa: D401
            return False

        def last_error(self):
            return (0, "MT5 not available")

        def account_info(self):
            return None

        def order_send(self, request):
            return None

        def shutdown(self):
            pass

    mt5 = _MT5Stub()

from loguru import logger

import sys
import os

# 支持从项目根目录或子模块直接运行时都能找到 Config
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import Config


class MT5Trader:
    """通过 MT5 Python API 执行市价订单的执行层。

    职责：
    - 连接 MT5 终端并验证账户可交易状态
    - 以市价单发起买入 / 卖出
    - 每次下单前查询账户净值与可用保证金
    - 将 Config.MAGIC_NUMBER 附加到所有订单
    """

    def __init__(self) -> None:
        self._connected: bool = False

    # ──────────────────────────────────────────────────────
    # 连接管理
    # ──────────────────────────────────────────────────────

    def connect(self) -> None:
        """连接 MT5 终端并验证账户可交易状态。

        Raises:
            ConnectionError: 若 mt5.initialize() 失败。
            RuntimeError: 若账户不可交易。
        """
        if not mt5.initialize():
            err = mt5.last_error()
            logger.error(f"MT5 initialize() failed: {err}")
            raise ConnectionError(f"MT5 connection failed: {err}")

        info = mt5.account_info()
        if info is None:
            logger.error("MT5 account_info() returned None after connect")
            raise RuntimeError("Unable to retrieve MT5 account info after connection")

        if hasattr(info, "trade_allowed") and not info.trade_allowed:
            logger.error(f"MT5 account {info.login} is not allowed to trade")
            raise RuntimeError(f"MT5 account {info.login} is not tradeable")

        self._connected = True
        logger.info(
            f"MT5 connected — login={getattr(info, 'login', '?')}, "
            f"server={getattr(info, 'server', '?')}, "
            f"balance={getattr(info, 'balance', '?')}"
        )

    def close(self) -> None:
        """释放 MT5 连接。"""
        mt5.shutdown()
        self._connected = False
        logger.info("MT5 connection closed")

    # ──────────────────────────────────────────────────────
    # 账户信息
    # ──────────────────────────────────────────────────────

    def get_account_info(self) -> dict | None:
        """查询账户净值与可用保证金。

        Returns:
            {"equity": float, "free_margin": float}，查询失败返回 None。
        """
        info = mt5.account_info()
        if info is None:
            logger.error("mt5.account_info() failed")
            return None
        return {
            "equity": float(info.equity),
            "free_margin": float(info.margin_free),
        }

    # ──────────────────────────────────────────────────────
    # 买入
    # ──────────────────────────────────────────────────────

    def buy(self, symbol: str, lot: float) -> bool:
        """发起 ORDER_TYPE_BUY 市价买单。

        流程：
        1. 查询账户信息，失败则中止并返回 False（Req 6.7）
        2. 构建 MqlTradeRequest（Req 6.2、6.6）
        3. 调用 mt5.order_send()
        4. retcode != 10009 时记录并返回 False（Req 6.5）

        Args:
            symbol: MT5 品种标识符，例如 "XAUUSD"。
            lot:    下单手数，支持小数以实现部分平仓（Req 6.4）。

        Returns:
            成功返回 True，否则返回 False。
        """
        # Step 1: 查询账户信息
        account = self.get_account_info()
        if account is None:
            logger.error(f"[buy] Aborted for {symbol}: account_info() failed")
            return False

        logger.info(
            f"[buy] {symbol} {lot} lot | "
            f"equity={account['equity']:.2f}, free_margin={account['free_margin']:.2f}"
        )

        # Step 2: 构建订单请求
        request = {
            "action":   mt5.TRADE_ACTION_DEAL,
            "symbol":   symbol,
            "volume":   float(lot),
            "type":     mt5.ORDER_TYPE_BUY,
            "magic":    Config.MAGIC_NUMBER,
            "comment":  "MT5AlphaGPT_buy",
            "type_time": 0,   # ORDER_TIME_GTC
            "type_filling": mt5.ORDER_FILLING_FOK,  # ORDER_FILLING_IOC（部分成交）
        }

        # Step 3: 发送订单
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"[buy] mt5.order_send() returned None for {symbol}")
            return False

        # Step 4: 检查 retcode
        retcode = result.retcode
        if retcode != 10009:
            comment = getattr(result, "comment", "")
            logger.error(
                f"[buy] Order rejected for {symbol}: retcode={retcode}, comment={comment}"
            )
            return False

        logger.info(f"[buy] Success — {symbol} {lot} lot, ticket={result.order}")
        return True

    # ──────────────────────────────────────────────────────
    # 卖出 / 平仓
    # ──────────────────────────────────────────────────────

    def sell(self, symbol: str, lot: float) -> bool:
        """发起 ORDER_TYPE_SELL 市价卖单（用于平多仓或做空）。

        支持通过指定 lot 进行部分平仓（Req 6.3、6.4）。

        Args:
            symbol: MT5 品种标识符。
            lot:    卖出手数。

        Returns:
            成功返回 True，否则返回 False。
        """
        # Step 1: 查询账户信息
        account = self.get_account_info()
        if account is None:
            logger.error(f"[sell] Aborted for {symbol}: account_info() failed")
            return False

        logger.info(
            f"[sell] {symbol} {lot} lot | "
            f"equity={account['equity']:.2f}, free_margin={account['free_margin']:.2f}"
        )

        # Step 2: 构建订单请求
        request = {
            "action":   mt5.TRADE_ACTION_DEAL,
            "symbol":   symbol,
            "volume":   float(lot),
            "type":     mt5.ORDER_TYPE_SELL,
            "magic":    Config.MAGIC_NUMBER,
            "comment":  "MT5AlphaGPT_sell",
            "type_time": 0,   # ORDER_TIME_GTC
            "type_filling": mt5.ORDER_FILLING_FOK,  # ORDER_FILLING_IOC
        }

        # Step 3: 发送订单
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"[sell] mt5.order_send() returned None for {symbol}")
            return False

        # Step 4: 检查 retcode
        retcode = result.retcode
        if retcode != 10009:
            comment = getattr(result, "comment", "")
            logger.error(
                f"[sell] Order rejected for {symbol}: retcode={retcode}, comment={comment}"
            )
            return False

        logger.info(f"[sell] Success — {symbol} {lot} lot, ticket={result.order}")
        return True

    # ──────────────────────────────────────────────────────
    # 精确平仓（按 ticket）
    # ──────────────────────────────────────────────────────

    def close_position(self, symbol: str, lot: float, direction: str,
                       ticket: int = 0) -> bool:
        """按仓位方向精确平仓。

        "BUY" 方向的多仓用 ORDER_TYPE_SELL 平；
        "SELL" 方向的空仓用 ORDER_TYPE_BUY 平。
        若 ticket > 0，则在请求里附加 position 字段，确保平的是指定订单，
        而不是盲目发反向市价单（避免误开新仓）。

        Args:
            symbol:    交易品种。
            lot:       平仓手数（支持部分平仓）。
            direction: 持仓方向，"BUY" 或 "SELL"。
            ticket:    MT5 订单号；0 表示不指定（仅在无法取到 ticket 时使用）。

        Returns:
            成功返回 True，否则返回 False。
        """
        account = self.get_account_info()
        if account is None:
            logger.error(f"[close_position] Aborted for {symbol}: account_info() failed")
            return False

        close_type = (mt5.ORDER_TYPE_SELL if direction == "BUY"
                      else mt5.ORDER_TYPE_BUY)
        comment    = (f"MT5AlphaGPT_close_long"
                      if direction == "BUY" else "MT5AlphaGPT_close_short")

        request: dict = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       float(lot),
            "type":         close_type,
            "magic":        Config.MAGIC_NUMBER,
            "comment":      comment,
            "type_time":    0,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        if ticket > 0:
            request["position"] = ticket   # 平指定 ticket，不误开新仓

        logger.info(
            f"[close_position] {symbol} {direction} lot={lot} ticket={ticket}"
        )
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"[close_position] order_send() returned None for {symbol}")
            return False
        if result.retcode != 10009:
            logger.error(
                f"[close_position] rejected: {symbol} retcode={result.retcode} "
                f"comment={getattr(result, 'comment', '')}"
            )
            return False
        logger.info(f"[close_position] Success — {symbol} ticket={result.order}")
        return True

    # ──────────────────────────────────────────────────────
    # 开空仓
    # ──────────────────────────────────────────────────────

    def open_short(self, symbol: str, lot: float) -> bool:
        """发起 ORDER_TYPE_SELL 开空仓（区别于 close_position 的平多）。

        使用独立的 comment 字段以便在 MT5 日志中区分开空与平多。

        Args:
            symbol: 交易品种。
            lot:    开仓手数。

        Returns:
            成功返回 True，否则返回 False。
        """
        account = self.get_account_info()
        if account is None:
            logger.error(f"[open_short] Aborted for {symbol}: account_info() failed")
            return False

        logger.info(
            f"[open_short] {symbol} {lot} lot | "
            f"equity={account['equity']:.2f}"
        )
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       float(lot),
            "type":         mt5.ORDER_TYPE_SELL,
            "magic":        Config.MAGIC_NUMBER,
            "comment":      "MT5AlphaGPT_open_short",
            "type_time":    0,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"[open_short] order_send() returned None for {symbol}")
            return False
        if result.retcode != 10009:
            logger.error(
                f"[open_short] rejected: {symbol} retcode={result.retcode} "
                f"comment={getattr(result, 'comment', '')}"
            )
            return False
        logger.info(f"[open_short] Success — {symbol} {lot} lot, ticket={result.order}")
        return True

    # ──────────────────────────────────────────────────────
    # 持仓查询 / 批量平仓
    # ──────────────────────────────────────────────────────

    def get_positions(
        self,
        symbol: str | None = None,
        magic: int | None = None,
    ) -> list:
        """查询 MT5 当前持仓，可按品种与 magic 过滤。"""
        if not _MT5_AVAILABLE or mt5 is None:
            return []
        if symbol:
            live = mt5.positions_get(symbol=symbol)
        else:
            live = mt5.positions_get()
        if live is None:
            return []
        positions = list(live)
        if magic is not None:
            positions = [p for p in positions if getattr(p, "magic", 0) == magic]
        return positions

    def close_all_positions(
        self,
        symbol: str,
        magic: int | None = None,
        *,
        filter_magic: bool = True,
    ) -> bool:
        """平掉指定品种下全部持仓（按 position ticket）。

        filter_magic=False 时平掉该品种所有 magic 的仓位。
        """
        eff_magic = None
        if filter_magic:
            eff_magic = Config.MAGIC_NUMBER if magic is None else magic
        ok_all = True

        while True:
            positions = self.get_positions(symbol, eff_magic)
            if not positions:
                return ok_all

            closed_any = False
            for p in positions:
                direction = "BUY" if getattr(p, "type", 0) == 0 else "SELL"
                ticket = int(getattr(p, "ticket", 0))
                lot = float(getattr(p, "volume", 0.0))
                if ticket <= 0 or lot <= 0:
                    continue
                if self.close_position(symbol, lot, direction, ticket):
                    closed_any = True
                    continue

                # 可能已被前一笔平掉；若 ticket 已不在持仓列表则视为成功
                still_open = {
                    int(getattr(x, "ticket", 0))
                    for x in self.get_positions(symbol, eff_magic)
                }
                if ticket not in still_open:
                    closed_any = True
                else:
                    ok_all = False

            if not closed_any:
                return ok_all
