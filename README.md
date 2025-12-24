# Intraday Trading Education

Binance Orderbook 분석을 통한 Intraday Trading 교육 프로젝트입니다.

## 학습 내용

1. **Orderbook (호가창)** - 매수/매도 주문 데이터 이해
2. **Bid-Ask Spread** - 유동성 지표 분석
3. **Mid-price vs Micro-price** - 가격 예측 지표 비교

## 설치

```bash
uv sync
```

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                         3가지 Runner                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │  ForwardRunner  │  │ OrderbookBack-  │  │ TickBacktest-   │      │
│  │  (실시간)       │  │ testRunner      │  │ Runner          │      │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘      │
│           │                    │                    │               │
│           └────────────────────┼────────────────────┘               │
│                                ▼                                    │
│           ┌─────────────────────────────────────────┐               │
│           │        공통 컴포넌트 (재사용)           │               │
│           │  • Strategy (Protocol)                 │               │
│           │  • PaperTrader                         │               │
│           │  • OrderbookProcessor                  │               │
│           │  • PerformanceCalculator               │               │
│           └─────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

## 실행

### 1. Tick 기반 백테스트 (Binance Public Data)

Binance에서 무료로 제공하는 히스토리컬 데이터를 다운로드하여 백테스트합니다.

```bash
# 스크립트로 실행
python scripts/run_tick_backtest.py
```

```python
# 코드로 직접 사용
from pathlib import Path
from intraday import (
    TickDataDownloader,
    TickDataLoader,
    TickBacktestRunner,
    BarType,
    OBIStrategy,
)

# 1. 데이터 다운로드 (한 번만)
downloader = TickDataDownloader()
downloader.download_monthly(
    symbol="BTCUSDT",
    year=2024,
    month=1,
    output_dir=Path("./data/ticks"),
)

# 2. 데이터 로드
loader = TickDataLoader(Path("./data/ticks"), symbol="BTCUSDT")

# 3. 전략 생성
strategy = OBIStrategy(
    buy_threshold=0.3,
    sell_threshold=-0.3,
    quantity=0.01,
)

# 4. 백테스트 실행
runner = TickBacktestRunner(
    strategy=strategy,
    data_loader=loader,
    bar_type=BarType.VOLUME,  # 볼륨바 (TICK, TIME도 가능)
    bar_size=10.0,            # 10 BTC마다 바 생성
    initial_capital=10000.0,
)

report = runner.run()
report.print_summary()
```

### 2. Orderbook 기반 백테스트 (직접 수집)

Binance는 오더북 히스토리를 제공하지 않으므로 직접 수집해야 합니다.

```bash
# Step 1: 오더북 데이터 수집 (1시간)
python scripts/record_orderbook.py --duration 3600

# Step 2: 백테스트 실행
python scripts/run_orderbook_backtest.py
```

```python
# 코드로 직접 사용
import asyncio
from pathlib import Path
from intraday import (
    OrderbookRecorder,
    OrderbookDataLoader,
    OrderbookBacktestRunner,
    OBIStrategy,
)

# 1. 오더북 수집 (비동기)
async def collect_data():
    recorder = OrderbookRecorder()
    await recorder.record(
        symbol="btcusdt",
        duration_seconds=3600,  # 1시간
        output_dir=Path("./data/orderbook"),
    )

asyncio.run(collect_data())

# 2. 백테스트 실행
loader = OrderbookDataLoader(Path("./data/orderbook"), symbol="btcusdt")
strategy = OBIStrategy(buy_threshold=0.3, sell_threshold=-0.3, quantity=0.01)

runner = OrderbookBacktestRunner(
    strategy=strategy,
    data_loader=loader,
    initial_capital=10000.0,
)

report = runner.run()
report.print_summary()
```

### 3. Forward Test (실시간)

실시간 WebSocket 데이터로 전략을 테스트합니다.

```bash
python scripts/run_forward_test.py
```

```python
import asyncio
from intraday import ForwardRunner, OBIStrategy

async def run_forward_test():
    strategy = OBIStrategy(buy_threshold=0.3, sell_threshold=-0.3)
    runner = ForwardRunner(
        strategy=strategy,
        symbol="btcusdt",
        initial_capital=10000.0,
    )
    
    await runner.run(duration_seconds=3600)  # 1시간 실행
    
    report = runner.get_performance_report()
    report.print_summary()

asyncio.run(run_forward_test())
```

### 4. Jupyter Notebook (학습용)

```bash
uv run jupyter notebook notebooks/01_orderbook_basics.ipynb
```

### 5. 실시간 대시보드

```bash
uv run python -m intraday.dashboard
```

## 세 가지 Runner 비교

| Runner | 데이터 소스 | 용도 |
|--------|------------|------|
| `TickBacktestRunner` | Binance Public Data (다운로드) | 과거 틱 데이터로 전략 검증 |
| `OrderbookBacktestRunner` | 직접 수집한 오더북 | 과거 OBI 전략 검증 |
| `ForwardRunner` | 실시간 WebSocket | 실시간 전략 테스트 |

## Bar 타입 (TickBacktestRunner)

| 타입 | 설명 | 예시 |
|-----|------|------|
| `BarType.VOLUME` | 거래량 기반 | `bar_size=10.0` → 10 BTC마다 바 생성 |
| `BarType.TICK` | 체결 횟수 기반 | `bar_size=100` → 100틱마다 바 생성 |
| `BarType.TIME` | 시간 기반 | `bar_size=60` → 60초마다 바 생성 |

## 빠른 시작 (권장 순서)

```bash
# 1. Tick 백테스트부터 시작 (데이터 즉시 다운로드 가능)
python scripts/run_tick_backtest.py

# 2. 오더북 데이터 수집 (백그라운드로 1시간)
python scripts/record_orderbook.py --duration 3600 &

# 3. 수집 완료 후 오더북 백테스트
python scripts/run_orderbook_backtest.py

# 4. 실시간 포워드 테스트
python scripts/run_forward_test.py
```

## 프로젝트 구조

```
src/intraday/
├── __init__.py
├── backtest/                  # 백테스터
│   ├── orderbook_runner.py    # 오더북 기반 백테스터
│   └── tick_runner.py         # 틱 기반 백테스터 (볼륨바/틱바)
├── data/                      # 데이터 수집/로딩
│   ├── downloader.py          # Binance Public Data 다운로더
│   ├── recorder.py            # 오더북 실시간 수집기
│   └── loader.py              # 데이터 로더
├── client.py                  # Binance WebSocket 클라이언트
├── orderbook.py               # 오더북 처리
├── paper_trader.py            # 가상 거래 시뮬레이터
├── performance.py             # 성과 분석
├── strategy.py                # 전략 인터페이스
└── runner.py                  # 포워드 테스트 러너

scripts/
├── run_tick_backtest.py       # Tick 백테스트 예제
├── run_orderbook_backtest.py  # Orderbook 백테스트 예제
├── record_orderbook.py        # 오더북 수집 스크립트
└── run_forward_test.py        # 포워드 테스트 예제
```

## 테스트

```bash
uv run pytest tests/ -v
```

## 환경 변수

`.env` 파일에 Binance API 키를 설정하세요 (선택사항, 공개 데이터는 API 키 불필요):

```
BINANCE_API_KEY=your_api_key_here
```
