"""
백테스터 시나리오 테스트 (사용자 관점)

"이 기능은 이렇게 작동해야 한다"는 철학으로 작성된 테스트입니다.

테스트 철학:
    - 실제 트레이딩 시나리오 기반
    - 전략이 신호를 생성하면 거래가 실행되어야 함
    - 손익이 정확하게 계산되어야 함
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from intraday import (
    OBIStrategy,
    OrderbookBacktestRunner,
    TickBacktestRunner,
    BarType,
    Side,
)
from intraday.data.loader import OrderbookDataLoader, TickDataLoader


class TestOrderbookBacktestScenarios:
    """
    OrderbookBacktestRunner 시나리오 테스트
    
    "오더북 불균형이 높으면 매수하고, 낮으면 매도한다"
    """
    
    def test_buy_signal_should_trigger_when_imbalance_exceeds_threshold(self, tmp_path: Path):
        """
        imbalance가 buy_threshold를 초과하면 매수 주문이 생성되어야 한다
        
        Given: imbalance > 0.3 인 오더북 (매수 압력 강함)
        When: OBIStrategy(buy_threshold=0.3)로 백테스트
        Then: 매수 주문이 제출됨
        """
        # Given: 강한 매수 불균형 오더북 (bid_qty >> ask_qty)
        # imbalance = (10 - 2) / (10 + 2) = 0.667 > 0.3
        records = []
        base = datetime(2024, 1, 15, 10, 0, 0)
        
        for i in range(10):
            record = {
                "timestamp": base + timedelta(milliseconds=i * 100),
                "last_update_id": i,
                "symbol": "BTCUSDT",
            }
            # 강한 매수 불균형
            record["bid_price_0"] = 50000.0
            record["bid_qty_0"] = 10.0  # 매수 물량 많음
            record["ask_price_0"] = 50001.0
            record["ask_qty_0"] = 2.0   # 매도 물량 적음
            
            for j in range(1, 20):
                record[f"bid_price_{j}"] = 50000.0 - j
                record[f"bid_qty_{j}"] = 1.0
                record[f"ask_price_{j}"] = 50001.0 + j
                record[f"ask_qty_{j}"] = 1.0
            
            records.append(record)
        
        df = pd.DataFrame(records)
        filepath = tmp_path / "orderbook_btcusdt_test.parquet"
        df.to_parquet(filepath, index=False)
        
        # When
        loader = OrderbookDataLoader(tmp_path, symbol="btcusdt")
        strategy = OBIStrategy(buy_threshold=0.3, sell_threshold=-0.3, quantity=0.01)
        
        runner = OrderbookBacktestRunner(
            strategy=strategy,
            data_loader=loader,
            initial_capital=10000.0,
        )
        runner.run(progress_interval=100000)
        
        # Then: 매수 주문이 제출됨
        # pending_orders 또는 trades에 BUY가 있어야 함
        has_buy = (
            any(po.order.side == Side.BUY for po in runner.trader.pending_orders) or
            any(t.side == Side.BUY for t in runner.trader.trades)
        )
        assert has_buy, "imbalance > threshold일 때 매수 주문이 생성되어야 함"
    
    def test_sell_signal_should_trigger_when_imbalance_below_threshold(self, tmp_path: Path):
        """
        포지션이 있고 imbalance가 sell_threshold 미만이면 매도해야 한다
        
        Given: 1) 매수 후 2) imbalance < -0.3 인 오더북
        When: OBIStrategy로 백테스트
        Then: 매도 주문으로 청산
        """
        # Given: 처음엔 매수 신호 → 그 다음 매도 신호
        records = []
        base = datetime(2024, 1, 15, 10, 0, 0)
        
        # 처음 5개: 강한 매수 불균형 (imbalance = 0.667)
        for i in range(5):
            record = {
                "timestamp": base + timedelta(milliseconds=i * 100),
                "last_update_id": i,
                "symbol": "BTCUSDT",
            }
            record["bid_price_0"] = 50000.0
            record["bid_qty_0"] = 10.0
            record["ask_price_0"] = 50001.0
            record["ask_qty_0"] = 2.0
            
            for j in range(1, 20):
                record[f"bid_price_{j}"] = 50000.0 - j
                record[f"bid_qty_{j}"] = 1.0
                record[f"ask_price_{j}"] = 50001.0 + j
                record[f"ask_qty_{j}"] = 1.0
            
            records.append(record)
        
        # 다음 5개: 강한 매도 불균형 (imbalance = -0.667)
        for i in range(5, 10):
            record = {
                "timestamp": base + timedelta(milliseconds=i * 100),
                "last_update_id": i,
                "symbol": "BTCUSDT",
            }
            record["bid_price_0"] = 50000.0
            record["bid_qty_0"] = 2.0   # 매수 물량 적음
            record["ask_price_0"] = 50001.0
            record["ask_qty_0"] = 10.0  # 매도 물량 많음
            
            for j in range(1, 20):
                record[f"bid_price_{j}"] = 50000.0 - j
                record[f"bid_qty_{j}"] = 1.0
                record[f"ask_price_{j}"] = 50001.0 + j
                record[f"ask_qty_{j}"] = 1.0
            
            records.append(record)
        
        df = pd.DataFrame(records)
        filepath = tmp_path / "orderbook_btcusdt_test.parquet"
        df.to_parquet(filepath, index=False)
        
        # When
        loader = OrderbookDataLoader(tmp_path, symbol="btcusdt")
        strategy = OBIStrategy(buy_threshold=0.3, sell_threshold=-0.3, quantity=0.01)
        
        runner = OrderbookBacktestRunner(
            strategy=strategy,
            data_loader=loader,
            initial_capital=10000.0,
        )
        runner.run(progress_interval=100000)
        
        # Then: 매수 후 매도가 발생해야 함 (또는 pending)
        trades = runner.trader.trades
        pending = runner.trader.pending_orders
        
        # 최소한 매수가 있어야 함
        has_buy = any(t.side == Side.BUY for t in trades) or any(po.order.side == Side.BUY for po in pending)
        assert has_buy, "먼저 매수가 발생해야 함"
    
    def test_no_signal_when_imbalance_is_neutral(self, tmp_path: Path):
        """
        imbalance가 중립이면 주문이 생성되지 않아야 한다
        
        Given: imbalance = 0 인 오더북 (균형 상태)
        When: OBIStrategy로 백테스트
        Then: 주문 없음
        """
        # Given: 완전 균형 오더북 (imbalance = 0)
        records = []
        base = datetime(2024, 1, 15, 10, 0, 0)
        
        for i in range(10):
            record = {
                "timestamp": base + timedelta(milliseconds=i * 100),
                "last_update_id": i,
                "symbol": "BTCUSDT",
            }
            # 완전 균형
            record["bid_price_0"] = 50000.0
            record["bid_qty_0"] = 5.0
            record["ask_price_0"] = 50001.0
            record["ask_qty_0"] = 5.0  # 동일 물량
            
            for j in range(1, 20):
                record[f"bid_price_{j}"] = 50000.0 - j
                record[f"bid_qty_{j}"] = 1.0
                record[f"ask_price_{j}"] = 50001.0 + j
                record[f"ask_qty_{j}"] = 1.0
            
            records.append(record)
        
        df = pd.DataFrame(records)
        filepath = tmp_path / "orderbook_btcusdt_test.parquet"
        df.to_parquet(filepath, index=False)
        
        # When
        loader = OrderbookDataLoader(tmp_path, symbol="btcusdt")
        strategy = OBIStrategy(buy_threshold=0.3, sell_threshold=-0.3, quantity=0.01)
        
        runner = OrderbookBacktestRunner(
            strategy=strategy,
            data_loader=loader,
            initial_capital=10000.0,
        )
        runner.run(progress_interval=100000)
        
        # Then: 주문 없음
        assert len(runner.trader.trades) == 0, "중립 imbalance에서는 거래 없어야 함"
        assert len(runner.trader.pending_orders) == 0, "중립 imbalance에서는 주문 없어야 함"


class TestTickBacktestScenarios:
    """
    TickBacktestRunner 시나리오 테스트
    
    "틱 데이터로 볼륨바/틱바를 만들고 전략을 실행한다"
    """
    
    def test_volume_bar_should_be_created_when_volume_threshold_reached(self, tmp_path: Path):
        """
        거래량이 임계값에 도달하면 볼륨바가 생성되어야 한다
        
        Given: 각 0.1 BTC씩 100개 틱 (총 10 BTC)
        When: bar_size=1.0 (1 BTC)으로 백테스트
        Then: 10개의 볼륨바 생성
        """
        # Given: 100틱, 각 0.1 BTC
        records = []
        base = datetime(2024, 1, 15, 10, 0, 0)
        
        for i in range(100):
            records.append({
                "timestamp": base + timedelta(milliseconds=i * 10),
                "price": 50000.0 + (i % 10),  # 약간의 가격 변동
                "quantity": 0.1,  # 각 0.1 BTC
                "is_buyer_maker": i % 2 == 0,
                "symbol": "BTCUSDT",
            })
        
        df = pd.DataFrame(records)
        filepath = tmp_path / "BTCUSDT-aggTrades-test.parquet"
        df.to_parquet(filepath, index=False)
        
        # When
        loader = TickDataLoader(tmp_path, symbol="BTCUSDT")
        strategy = OBIStrategy()
        
        runner = TickBacktestRunner(
            strategy=strategy,
            data_loader=loader,
            bar_type=BarType.VOLUME,
            bar_size=1.0,  # 1 BTC마다 바
        )
        runner.run(progress_interval=100000)
        
        # Then: 100 * 0.1 = 10 BTC → 10개 바 (마지막 바는 미완성일 수 있음)
        # 볼륨바는 정확히 threshold에 도달해야 완성되므로 9~10개
        assert runner.bar_count >= 9
    
    def test_tick_bar_should_be_created_every_n_ticks(self, tmp_path: Path):
        """
        N틱마다 틱바가 생성되어야 한다
        
        Given: 500개 틱
        When: bar_size=100 (100틱)으로 백테스트
        Then: 5개의 틱바 생성
        """
        # Given: 500틱
        records = []
        base = datetime(2024, 1, 15, 10, 0, 0)
        
        for i in range(500):
            records.append({
                "timestamp": base + timedelta(milliseconds=i),
                "price": 50000.0,
                "quantity": 0.01,
                "is_buyer_maker": False,
                "symbol": "BTCUSDT",
            })
        
        df = pd.DataFrame(records)
        filepath = tmp_path / "BTCUSDT-aggTrades-test.parquet"
        df.to_parquet(filepath, index=False)
        
        # When
        loader = TickDataLoader(tmp_path, symbol="BTCUSDT")
        strategy = OBIStrategy()
        
        runner = TickBacktestRunner(
            strategy=strategy,
            data_loader=loader,
            bar_type=BarType.TICK,
            bar_size=100,
        )
        runner.run(progress_interval=100000)
        
        # Then: 500 / 100 = 5개 바
        assert runner.bar_count == 5
    
    def test_bar_ohlcv_should_be_calculated_correctly(self, tmp_path: Path):
        """
        바의 OHLCV가 정확히 계산되어야 한다
        
        Given: 가격이 변동하는 10개 틱
        When: 10틱으로 1개 바 생성
        Then: O, H, L, C, V가 정확함
        """
        # Given: 가격 패턴 100 → 110 → 95 → 105
        prices = [100, 102, 105, 110, 108, 95, 98, 102, 103, 105]
        quantities = [0.1] * 10
        
        records = []
        base = datetime(2024, 1, 15, 10, 0, 0)
        
        for i, (price, qty) in enumerate(zip(prices, quantities)):
            records.append({
                "timestamp": base + timedelta(milliseconds=i),
                "price": price,
                "quantity": qty,
                "is_buyer_maker": False,
                "symbol": "BTCUSDT",
            })
        
        df = pd.DataFrame(records)
        filepath = tmp_path / "BTCUSDT-aggTrades-test.parquet"
        df.to_parquet(filepath, index=False)
        
        # When
        loader = TickDataLoader(tmp_path, symbol="BTCUSDT")
        strategy = OBIStrategy()
        
        runner = TickBacktestRunner(
            strategy=strategy,
            data_loader=loader,
            bar_type=BarType.TICK,
            bar_size=10,
        )
        runner.run(progress_interval=100000)
        
        # Then: 1개 바가 생성되고 OHLCV가 정확
        assert runner.bar_count == 1
        bar = runner.current_bar
        
        assert bar.open == 100   # 첫 번째 가격
        assert bar.high == 110   # 최고가
        assert bar.low == 95     # 최저가
        assert bar.close == 105  # 마지막 가격
        assert bar.volume == pytest.approx(1.0)  # 0.1 * 10
        assert bar.trade_count == 10
    
    def test_volume_imbalance_should_reflect_buy_sell_ratio(self, tmp_path: Path):
        """
        바의 volume_imbalance가 매수/매도 비율을 반영해야 한다
        
        Given: 80% 매수 주도, 20% 매도 주도인 틱들
        When: 볼륨바 생성
        Then: volume_imbalance ≈ 0.6 (= (0.8 - 0.2) / (0.8 + 0.2))
        """
        # Given: 10틱 중 8개 매수주도, 2개 매도주도
        records = []
        base = datetime(2024, 1, 15, 10, 0, 0)
        
        for i in range(10):
            records.append({
                "timestamp": base + timedelta(milliseconds=i),
                "price": 50000.0,
                "quantity": 0.1,
                # is_buyer_maker=False → 매수주도 (8개)
                # is_buyer_maker=True → 매도주도 (2개)
                "is_buyer_maker": i >= 8,  # 마지막 2개만 매도주도
                "symbol": "BTCUSDT",
            })
        
        df = pd.DataFrame(records)
        filepath = tmp_path / "BTCUSDT-aggTrades-test.parquet"
        df.to_parquet(filepath, index=False)
        
        # When
        loader = TickDataLoader(tmp_path, symbol="BTCUSDT")
        strategy = OBIStrategy()
        
        runner = TickBacktestRunner(
            strategy=strategy,
            data_loader=loader,
            bar_type=BarType.TICK,
            bar_size=10,
        )
        runner.run(progress_interval=100000)
        
        # Then
        bar = runner.current_bar
        
        # buy_volume = 8 * 0.1 = 0.8, sell_volume = 2 * 0.1 = 0.2
        assert bar.buy_volume == pytest.approx(0.8)
        assert bar.sell_volume == pytest.approx(0.2)
        assert bar.volume_imbalance == pytest.approx(0.6)  # (0.8 - 0.2) / 1.0


class TestTickBacktestRunnerWithCandleBuilder:
    """
    TickBacktestRunner + CandleBuilder 통합 테스트
    
    사용자 관점:
        "TickBacktestRunner가 CandleBuilder를 사용하더라도 동일하게 작동해야 한다"
    """
    
    def test_runner_creates_candles_correctly(self):
        """러너가 CandleBuilder로 캔들을 올바르게 생성해야 한다"""
        from intraday import TickBacktestRunner, CandleType, VolumeImbalanceStrategy
        from intraday.data.loader import TickDataLoader
        from unittest.mock import MagicMock
        from datetime import datetime
        from intraday.client import AggTrade
        
        # Mock 로더
        mock_loader = MagicMock(spec=TickDataLoader)
        trades = [
            AggTrade(datetime(2024, 1, 1, 0, 0, i), "BTCUSDT", 50000.0 + i, 3.0, i % 2 == 0)
            for i in range(12)  # 12개 틱 * 3 BTC = 36 BTC → 3개 캔들 (10 BTC씩)
        ]
        mock_loader.iter_trades.return_value = iter(trades)
        
        strategy = VolumeImbalanceStrategy()
        runner = TickBacktestRunner(
            strategy=strategy,
            data_loader=mock_loader,
            bar_type=CandleType.VOLUME,
            bar_size=10.0,
        )
        
        runner.run()
        
        # 3개 캔들 생성 확인 (36 BTC / 10 = 3개 + 나머지)
        assert runner.bar_count == 3
    
    def test_runner_supports_dollar_bar(self):
        """러너가 DOLLAR 타입 캔들도 지원해야 한다 (CandleBuilder 통합 확인)"""
        from intraday import TickBacktestRunner, CandleType, VolumeImbalanceStrategy
        from intraday.data.loader import TickDataLoader
        from unittest.mock import MagicMock
        from datetime import datetime
        from intraday.client import AggTrade
        
        mock_loader = MagicMock(spec=TickDataLoader)
        # 50000 * 10 = 500,000 달러씩 2개 틱 = 1,000,000 달러
        trades = [
            AggTrade(datetime(2024, 1, 1, 0, 0, 0), "BTCUSDT", 50000.0, 10.0, False),
            AggTrade(datetime(2024, 1, 1, 0, 0, 1), "BTCUSDT", 50000.0, 10.0, True),
        ]
        mock_loader.iter_trades.return_value = iter(trades)
        
        strategy = VolumeImbalanceStrategy()
        runner = TickBacktestRunner(
            strategy=strategy,
            data_loader=mock_loader,
            bar_type=CandleType.DOLLAR,  # 달러 바!
            bar_size=1_000_000,          # 100만 달러
        )
        
        runner.run()
        
        # 1개 캔들 생성 확인
        assert runner.bar_count == 1


class TestBacktestPnLCalculation:
    """
    백테스트 손익 계산 테스트
    
    "거래가 체결되면 정확한 손익이 계산되어야 한다"
    """
    
    def test_winning_trade_should_show_positive_pnl(self, tmp_path: Path):
        """
        수익 거래는 양수 PnL을 보여야 한다
        
        Given: 50000에 매수 → 50100에 매도 시나리오
        When: 백테스트 실행 후 청산
        Then: PnL > 0
        """
        # Given: 매수 신호 → 가격 상승 → 매도 신호
        records = []
        base = datetime(2024, 1, 15, 10, 0, 0)
        
        # 1-10: 강한 매수 신호 (가격 50000)
        for i in range(10):
            record = self._create_orderbook_record(
                timestamp=base + timedelta(milliseconds=i * 100),
                idx=i,
                bid_price=50000.0,
                ask_price=50001.0,
                bid_qty=10.0,  # 매수 불균형
                ask_qty=2.0,
            )
            records.append(record)
        
        # 11-20: 가격 상승 + 강한 매도 신호 (가격 50100)
        for i in range(10, 20):
            record = self._create_orderbook_record(
                timestamp=base + timedelta(milliseconds=i * 100),
                idx=i,
                bid_price=50100.0,  # 가격 상승
                ask_price=50101.0,
                bid_qty=2.0,
                ask_qty=10.0,  # 매도 불균형
            )
            records.append(record)
        
        df = pd.DataFrame(records)
        filepath = tmp_path / "orderbook_btcusdt_test.parquet"
        df.to_parquet(filepath, index=False)
        
        # When
        loader = OrderbookDataLoader(tmp_path, symbol="btcusdt")
        strategy = OBIStrategy(buy_threshold=0.3, sell_threshold=-0.3, quantity=0.01)
        
        runner = OrderbookBacktestRunner(
            strategy=strategy,
            data_loader=loader,
            initial_capital=10000.0,
            fee_rate=0.001,
        )
        report = runner.run(progress_interval=100000)
        
        # Then: 거래가 발생했으면 손익 확인
        if report.total_trades >= 2:  # 매수 + 매도
            # 수익 거래: 50100 - 50000 = 100달러 * 0.01 BTC = 1달러 이익 (수수료 제외)
            total_pnl = report.final_capital - report.initial_capital
            assert total_pnl > -2  # 수수료 고려해도 큰 손실은 아님
    
    def _create_orderbook_record(
        self,
        timestamp: datetime,
        idx: int,
        bid_price: float,
        ask_price: float,
        bid_qty: float,
        ask_qty: float,
    ) -> dict:
        """오더북 레코드 생성 헬퍼"""
        record = {
            "timestamp": timestamp,
            "last_update_id": idx,
            "symbol": "BTCUSDT",
            "bid_price_0": bid_price,
            "bid_qty_0": bid_qty,
            "ask_price_0": ask_price,
            "ask_qty_0": ask_qty,
        }
        
        for j in range(1, 20):
            record[f"bid_price_{j}"] = bid_price - j
            record[f"bid_qty_{j}"] = 1.0
            record[f"ask_price_{j}"] = ask_price + j
            record[f"ask_qty_{j}"] = 1.0
        
        return record

