from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple


GAMMA_BASE = "https://gamma-api.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

WALLET_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
CONDITION_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")


@dataclass
class AnalyzerConfig:
    out_dir: str = "out"
    sleep_between_requests: float = 0.05
    timeout_seconds: int = 30
    max_retries: int = 5

    trades_page_limit: int = 1_000
    trades_max_offset: int = 3_000
    positions_page_limit: int = 500
    positions_max_offset: int = 10_000
    closed_page_limit: int = 50
    closed_max_offset: int = 100_000

    with_clv: bool = False
    clv_horizons_seconds: Tuple[int, ...] = (3600, 86400)
    clv_fidelity_minutes: int = 60
    max_clv_assets_per_user: int = 250

    min_copy_shares: float = 10.0
    min_copy_value_usdc: float = 10.0
    min_copy_liquidity_usdc: float = 100.0
    max_copy_price_worse_than_avg: float = 0.05
    max_copy_spread: float = 0.05
    recent_buy_days: float = 30.0
