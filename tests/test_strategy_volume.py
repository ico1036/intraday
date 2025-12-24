"""
VolumeImbalanceStrategy 테스트

사용자 관점:
    "틱 데이터의 매수/매도 주도 비율을 기반으로 매매 신호를 생성해야 한다"
"""

from datetime import datetime

import pytest

from intraday import VolumeImbalanceStrategy, Side, MarketState


def make_state(
    imbalance: float,
    position_side: Side | None = None,
    position_qty: float = 0.0,
    mid_price: float = 50000.0,
) -> MarketState:
    """테스트용 MarketState 생성"""
    return MarketState(
        timestamp=datetime(2024, 1, 1),
        mid_price=mid_price,
        imbalance=imbalance,
        spread=1.0,
        spread_bps=0.02,
        best_bid=mid_price - 0.5,
        best_ask=mid_price + 0.5,
        best_bid_qty=10.0,
        best_ask_qty=10.0,
        position_side=position_side,
        position_qty=position_qty,
    )


class TestVolumeImbalanceStrategy:
    """VolumeImbalanceStrategy 테스트"""
    
    def test_default_initialization(self):
        """기본값으로 초기화되어야 한다"""
        strategy = VolumeImbalanceStrategy()
        
        assert strategy.buy_threshold == 0.4
        assert strategy.sell_threshold == -0.4
        assert strategy.quantity == 0.01
    
    def test_custom_initialization(self):
        """커스텀 값으로 초기화되어야 한다"""
        strategy = VolumeImbalanceStrategy(
            buy_threshold=0.5,
            sell_threshold=-0.5,
            quantity=0.05,
        )
        
        assert strategy.buy_threshold == 0.5
        assert strategy.sell_threshold == -0.5
        assert strategy.quantity == 0.05
    
    def test_buy_signal_when_volume_imbalance_high(self):
        """볼륨 불균형이 높으면(매수 주도) BUY 신호가 나와야 한다"""
        strategy = VolumeImbalanceStrategy(buy_threshold=0.4)
        
        # TickBacktestRunner가 volume_imbalance를 imbalance 필드에 넣음
        state = make_state(imbalance=0.5)
        
        order = strategy.generate_order(state)
        
        assert order is not None
        assert order.side == Side.BUY
    
    def test_sell_signal_when_volume_imbalance_low(self):
        """볼륨 불균형이 낮으면(매도 주도) 포지션이 있을 때 SELL 신호"""
        strategy = VolumeImbalanceStrategy(sell_threshold=-0.4)
        
        # BUY 포지션이 있는 상태
        state = make_state(imbalance=-0.5, position_side=Side.BUY, position_qty=0.01)
        
        order = strategy.generate_order(state)
        
        assert order is not None
        assert order.side == Side.SELL
    
    def test_no_signal_when_imbalance_neutral(self):
        """볼륨 불균형이 중립이면 신호 없음"""
        strategy = VolumeImbalanceStrategy(buy_threshold=0.4, sell_threshold=-0.4)
        
        state = make_state(imbalance=0.0)
        
        order = strategy.generate_order(state)
        
        assert order is None
    
    def test_no_sell_without_position(self):
        """포지션 없이는 SELL 신호가 나오면 안 된다 (현물)"""
        strategy = VolumeImbalanceStrategy(sell_threshold=-0.4)
        
        # 강한 매도 주도지만 포지션 없음
        state = make_state(imbalance=-0.6, position_side=None)
        
        order = strategy.generate_order(state)
        
        # 현물에서는 포지션 없이 공매도 불가
        assert order is None
    
    def test_no_duplicate_buy(self):
        """이미 BUY 포지션이면 추가 BUY 불가"""
        strategy = VolumeImbalanceStrategy(buy_threshold=0.4)
        
        # 이미 BUY 포지션 있음
        state = make_state(imbalance=0.6, position_side=Side.BUY, position_qty=0.01)
        
        order = strategy.generate_order(state)
        
        assert order is None
