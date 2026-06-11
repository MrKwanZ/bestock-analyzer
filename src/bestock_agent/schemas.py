"""Pydantic output schemas shared across all graph nodes."""

from datetime import date
from enum import Enum

from pydantic import BaseModel, EmailStr, Field


# ── Enumerations ──────────────────────────────────────────────────────────────


class TrendLabel(str, Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    PULLBACK = "pullback"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ChartType(str, Enum):
    PRICE_TREND = "price_trend"
    DAILY_CHANGE = "daily_change"
    INDEX_COMPARISON = "index_comparison"
    SENTIMENT = "sentiment"


class ErrorType(str, Enum):
    NETWORK_ERROR = "network_error"
    TOOL_ERROR = "tool_error"
    RATE_LIMIT = "rate_limit"
    VALIDATION_ERROR = "validation_error"
    TOKEN_LIMIT = "token_limit"
    UNKNOWN = "unknown"


# ── Market data ───────────────────────────────────────────────────────────────


class TopGainer(BaseModel):
    """Best-performing NASDAQ stock for the target day."""

    symbol: str
    name: str
    price: float = Field(gt=0)
    change: float
    change_pct: float
    market_cap: float | None = None
    volume: int | None = None


class PriceBar(BaseModel):
    """OHLCV bar for a single trading day."""

    date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)
    change_pct: float | None = None


# ── Analysis ──────────────────────────────────────────────────────────────────


class TrendAnalysis(BaseModel):
    """Quantitative trend summary over the lookback window."""

    symbol: str
    bars: list[PriceBar]
    daily_changes: list[float]
    avg_change: float
    trend_label: TrendLabel
    volatility: float | None = None
    summary: str = ""


class SentimentDriver(BaseModel):
    text: str
    sentiment: SentimentLabel
    source: str


class SentimentResult(BaseModel):
    """LLM-derived sentiment analysis from news headlines."""

    symbol: str
    score: float = Field(ge=-1.0, le=1.0, description="Normalised score: -1 (very negative) to +1 (very positive)")
    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    drivers: list[SentimentDriver] = []
    sources: list[str] = []
    summary: str = ""


class IndexBar(BaseModel):
    date: date
    close: float = Field(gt=0)
    change_pct: float


class IndexData(BaseModel):
    symbol: str
    name: str
    bars: list[IndexBar]
    period_change_pct: float


class IndexComparison(BaseModel):
    """Performance of the selected stock relative to market indices."""

    stock_symbol: str
    sp500: IndexData
    dow_jones: IndexData
    relative_perf_vs_sp500: float
    relative_perf_vs_dow: float
    beta: float | None = None
    excess_return: float | None = None


# ── Output artefacts ──────────────────────────────────────────────────────────


class ChartArtifact(BaseModel):
    path: str
    chart_type: ChartType
    title: str
    format: str = "png"


class EmailPayload(BaseModel):
    recipient: EmailStr
    subject: str
    body_html: str
    body_text: str
    chart_paths: list[str] = []


# ── Error & run summary ───────────────────────────────────────────────────────


class AgentError(BaseModel):
    error_type: ErrorType
    message: str
    node: str
    timestamp: str
    recoverable: bool = True
    fallback_used: bool = False


class RunSummary(BaseModel):
    success: bool
    stock_symbol: str | None = None
    stock_name: str | None = None
    change_pct: float | None = None
    email_sent: bool = False
    email_recipient: str | None = None
    errors: list[AgentError] = []
    charts_generated: int = 0
    message: str = ""
