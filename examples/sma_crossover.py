"""
Example: SMA Crossover Strategy

This demonstrates a correctly implemented trading strategy for Backtrader.
Use this as a reference for the expected output format.
"""

import backtrader as bt


class TradingStrategy(bt.Strategy):
    """
    Simple Moving Average Crossover Strategy.
    
    Buy when fast SMA crosses above slow SMA.
    Sell when fast SMA crosses below slow SMA.
    """
    
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
    )
    
    def __init__(self):
        # Initialize indicators
        self.fast_sma = bt.indicators.SMA(
            self.data.close, 
            period=self.params.fast_period
        )
        self.slow_sma = bt.indicators.SMA(
            self.data.close, 
            period=self.params.slow_period
        )
        
        # Crossover indicator for signal detection
        self.crossover = bt.indicators.CrossOver(
            self.fast_sma, 
            self.slow_sma
        )
        
        # Track pending orders
        self.order = None
    
    def next(self):
        # Skip if we have a pending order
        if self.order:
            return
        
        # Check for entry/exit signals
        if not self.position:
            # No position - check for buy signal
            if self.crossover > 0:
                self.order = self.buy()
        else:
            # Have position - check for sell signal
            if self.crossover < 0:
                self.order = self.sell()
    
    def notify_order(self, order):
        # Reset order tracker when order is complete
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None


if __name__ == "__main__":
    # Quick test
    import yfinance as yf
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TradingStrategy)
    
    # Load sample data
    df = yf.Ticker("AAPL").history(start="2023-01-01", end="2024-01-01")
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    
    # Set initial capital
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.001)
    
    # Add analyzer
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    print(f"Starting value: ${cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    print(f"Ending value: ${cerebro.broker.getvalue():.2f}")
    
    # Print trade count
    strat = results[0]
    trades = strat.analyzers.trades.get_analysis()
    total = trades.get('total', {}).get('total', 0) if trades else 0
    print(f"Total trades: {total}")

