#!/usr/bin/env python3
"""
Orderbook 데이터 수집 스크립트

실시간 WebSocket으로 오더북 스냅샷을 수집하여 Parquet로 저장합니다.

사용법:
    # 1분 동안 수집 (테스트용)
    python scripts/record_orderbook.py --duration 60
    
    # 1시간 동안 수집
    python scripts/record_orderbook.py --duration 3600
    
    # 1일 동안 수집
    python scripts/record_orderbook.py --duration 86400

교육 포인트:
    - Binance는 오더북 히스토리를 제공하지 않음
    - 백테스트용 오더북 데이터는 직접 수집해야 함
    - 100ms 간격으로 하루 약 864,000개 스냅샷
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intraday import OrderbookRecorder


async def main():
    """오더북 수집 메인 함수"""
    
    # 인자 파싱
    parser = argparse.ArgumentParser(description="Orderbook 데이터 수집")
    parser.add_argument(
        "--symbol",
        type=str,
        default="btcusdt",
        help="거래쌍 (기본: btcusdt)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="수집 시간 (초, 기본: 60)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/orderbook",
        help="출력 디렉토리 (기본: ./data/orderbook)",
    )
    parser.add_argument(
        "--include-trades",
        action="store_true",
        help="체결 데이터도 함께 수집",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Orderbook 데이터 수집")
    print("=" * 60)
    print(f"심볼: {args.symbol.upper()}")
    print(f"수집 시간: {args.duration}초")
    print(f"출력 디렉토리: {args.output}")
    print(f"체결 데이터 포함: {args.include_trades}")
    print("=" * 60)
    
    # 수집기 생성
    recorder = OrderbookRecorder(
        depth_levels=20,
        flush_interval=10000,
    )
    
    # 수집 시작
    try:
        ob_filepath, trade_filepath = await recorder.record(
            symbol=args.symbol,
            duration_seconds=args.duration,
            output_dir=Path(args.output),
            include_trades=args.include_trades,
        )
        
        print("\n" + "=" * 60)
        print("수집 완료!")
        print(f"오더북 파일: {ob_filepath}")
        if trade_filepath:
            print(f"체결 파일: {trade_filepath}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
        await recorder.stop()


if __name__ == "__main__":
    asyncio.run(main())

