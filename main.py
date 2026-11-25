"""Tiny dip-buying bot for make.fun bars-replay/v1 (agent-fun/bars-replay/v1).

Paper-trades hourly Solana bars. One file, standard library only, no network,
no clock, no randomness: the same input always produces the same orders, so
two replays of the bot agree byte for byte.

How it trades
------------
The window this bounty scores on is three days of hourly bars, and the dev
window it practises on falls the whole way: every symbol ends lower than it
starts. Buying and holding that window loses about 8.7%. So this bot does
not hold through a slide. It waits for a dip, buys the bounce, and gets out:

* Entry: once every 4 bars, rank the symbols whose latest close sits more
  than 1.5% below their 12-bar exponential average (the dip), deepest dip
  first, and buy up to 3 of them, at most 25% of equity each.
* Exit: sell the whole position when the close climbs back above the
  12-bar average, when the close is 3% above the fill price, or when 24
  bars have passed since the fill, whichever comes first.
* It never sells and buys in the same reply: a sell's proceeds only fund a
  buy on the next bar.

Wire protocol (hello / batch / orders / bar)
---------------------------------------------
The harness sends one JSON line at a time on stdin: a hello line first,
then one batch line per bar, then a done line. Each batch carries the bars
for that hour, the fills of the orders decided on the previous bar, and the
account. The bot answers every batch with exactly one reply line echoing
its seq and the orders decided on that bar. Orders decided on bar k fill
at bar k+1's open, so every decision here uses closes the bot has already
seen: no lookahead.
"""

import json
import sys

PROTOCOL = "agent-fun/bars-replay/v1"

# Strategy constants (fixed numbers, chosen from the dev window and kept
# deliberately plain so they survive a different three-day window).
EMA_SPAN = 12            # bars in the dip average
DIP_BPS = 150            # buy when close is this far below the average (bps)
TAKE_BPS = 300           # sell when close is this far above the fill (bps)
TOP_N = 3                # most symbols held at once
REBALANCE_EVERY = 4      # bars between entry scans
MAX_HOLD_BARS = 24       # longest a position is kept
MAX_WEIGHT_BPS = 2500    # most of equity spent on one symbol (the cap itself)
MAX_SLIPPAGE_BPS = 500   # widest move from the latest close a fill may take
MIN_NOTIONAL = 1000000   # smallest buy, in quote units (from hello fees)


class Bot:
    """Keeps the replay state: closes, the dip average, and open positions."""

    def __init__(self, hello):
        self.symbols = [s["symbol"] for s in hello["symbols"]]
        self.caps = hello["caps"]
        self.min_notional = int(hello["fees"]["min_notional_quote_units"])
        self.warmup_bars = hello.get("warmup_bars", 0)
        self.cash = int(hello["capital"]["quote_units"])
        self.equity = self.cash
        self.positions = {}       # symbol -> qty (int, from the account)
        self.closes = {s: [] for s in self.symbols}
        self.ema = {}             # symbol -> 12-bar exponential average
        self.entry = {}           # symbol -> fill price of the open position
        self.entry_bar = {}       # symbol -> batch seq the buy filled on
        self.peak = {}            # symbol -> highest close since the fill
        self.next_id = 1

    def _client_id(self):
        cid = "o%d" % self.next_id
        self.next_id += 1
        return cid

    def on_batch(self, batch):
        """Decide the orders for one batch of bars; returns the order list."""
        seq = batch["seq"]
        # Fills first: a buy that filled on this bar opens a position, and
        # its fill price is the entry the exits measure from.
        for fill in batch["fills"]:
            if fill["status"] != "filled":
                continue
            symbol = fill["symbol"]
            if fill["side"] == "buy":
                self.entry[symbol] = int(fill["price"])
                self.entry_bar[symbol] = seq
                self.peak[symbol] = int(fill["price"])
        # Then the account, then the new closes and the dip average.
        account = batch["account"]
        self.cash = int(account["cash_quote_units"])
        self.equity = int(account["equity_quote_units"])
        self.positions = {
            p["symbol"]: int(p["qty"]) for p in account["positions"]
        }
        bars = {b["symbol"]: b for b in batch["bars"]}
        for symbol in self.symbols:
            bar = bars.get(symbol)
            if bar is None:
                continue
            close = int(bar["c"])
            self.closes[symbol].append(close)
            if symbol in self.ema:
                k = 2.0 / (EMA_SPAN + 1)
                self.ema[symbol] = close * k + self.ema[symbol] * (1.0 - k)
            else:
                self.ema[symbol] = float(close)
            if symbol in self.peak and close > self.peak[symbol]:
                self.peak[symbol] = close
        if seq < self.warmup_bars:
            return []
        # Exits first: a symbol leaves when it bounces back over its
        # average, pays the take-profit, or simply runs out of time.
        sells = []
        for symbol in self.symbols:
            if symbol not in self.positions or symbol not in self.entry:
                continue
            close = self.closes[symbol][-1]
            entry = self.entry[symbol]
            bounced = close >= self.ema[symbol]
            took_profit = close * 10000 >= entry * (10000 + TAKE_BPS)
            timed_out = seq - self.entry_bar[symbol] >= MAX_HOLD_BARS
            if bounced or took_profit or timed_out:
                sells.append({
                    "side": "sell",
                    "client_id": self._client_id(),
                    "symbol": symbol,
                    "qty": str(self.positions[symbol]),
                    "max_slippage_bps": MAX_SLIPPAGE_BPS,
                })
        if sells:
            return sells
        # Entries: every fourth bar, deepest dips first.
        if seq % REBALANCE_EVERY != 0 or seq < EMA_SPAN * 2:
            return []
        slots = TOP_N - len(self.positions)
        if slots <= 0:
            return []
        dips = []
        for symbol in self.symbols:
            if symbol in self.positions or symbol not in self.ema:
                continue
            close = self.closes[symbol][-1]
            average = self.ema[symbol]
            if close * 10000 < average * (10000 - DIP_BPS):
                dips.append((average / close - 1.0, symbol))
        dips.sort(key=lambda item: (-item[0], item[1]))
        buys = []
        cash_left = self.cash
        for _, symbol in dips[:slots]:
            room = MAX_WEIGHT_BPS * self.equity // 10000
            spend = min(cash_left // max(1, min(len(dips), slots)), room)
            if spend < self.min_notional or spend <= 0:
                continue
            buys.append({
                "side": "buy",
                "client_id": self._client_id(),
                "symbol": symbol,
                "notional_quote_units": str(spend),
                "max_slippage_bps": MAX_SLIPPAGE_BPS,
            })
            cash_left -= spend
        return buys


def main():
    """Speak the replay protocol until the done line, then stop."""
    bot = None
    out = sys.stdout
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        kind = msg.get("type")
        if kind == "hello":
            if msg.get("protocol") != PROTOCOL:
                raise SystemExit("unexpected protocol: %r" % msg.get("protocol"))
            bot = Bot(msg)
        elif kind == "batch":
            orders = bot.on_batch(msg)
            reply = json.dumps(
                {"type": "reply", "seq": msg["seq"], "orders": orders},
                separators=(",", ":"),
            )
            out.write(reply + "\n")
            out.flush()
        elif kind == "done":
            return


if __name__ == "__main__":
    main()
