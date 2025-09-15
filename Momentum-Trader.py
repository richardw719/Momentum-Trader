# Import necessary libraries
import yfinance as yf
import pandas as pd
import backtrader as bt
from backtrader.indicators import RSI, BollingerBands, MovingAverageSimple, CrossOver
import matplotlib.pyplot as plt
import datetime


class SwingTrade(bt.Strategy):
    # Relevant Parameters for our strategy
    params = (
        ('rsi_period', 14),
        ('overbought', 70),
        ('rsi_neutral', 50),
        ('bb_period', 20),
        ('bb_std', 2),
        ('pfast', 20),
        ('pslow', 50)
    )



    # Log our trades
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')



    # Initialization
    def __init__(self):
        self.dataclose = self.datas[0].close

        self.order = None

        self.entry_price = None

        self.slow_sma = MovingAverageSimple(self.datas[0],
                                            period = self.params.pslow)

        self.fast_sma = MovingAverageSimple(self.datas[0],
                                            period=self.params.pfast)

        self.crossover = CrossOver(self.fast_sma, self.slow_sma)

        self.rsi = RSI(period = self.params.rsi_period)

        self.bbands = BollingerBands(period = self.params.bb_period,
                                     devfactor = self.params.bb_std)

        self.avg_volume = MovingAverageSimple(self.datas[0].volume, period=10)


    # Streamline the Orders
    def notify_order(self, order):
        # Check if order is submitted/accepted
        if order.status in [order.Submitted, order.Accepted]:
            return

        # Check if order is completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, {order.executed.price:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Reset order
        self.order = None



    # Main Logic for Signal Generation
    def next(self):
        # Check for any current open orders
        if self.order:
             return

        # Check if currently in the market
        if not self.position:
            # BUY if weighted signal score exceeds min threshold
            signal_score = 0

            # SMA crossover
            if self.crossover > 0:
                signal_score += 1

            # RSI
            if self.rsi[0] < self.params.rsi_neutral:
                pct_change = ((self.params.rsi_neutral -
                              self.rsi[0])/self.params.rsi_neutral)
                signal_score += pct_change

            # Bollinger Bands
            if (self.dataclose[0] <= self.bbands.lines.mid[0]):
                pct_change = (self.bbands.lines.mid[0] -
                              self.dataclose[0])/(self.bbands.lines.mid[0] -
                                                  self.bbands.lines.bot[0])
                signal_score += pct_change

            # Volume Metric
            if (self.datas[0].volume > 1.5 * self.avg_volume):
                signal_score += 0.1

            # Generate BUY signal
            if signal_score >= 1.15:
                self.log(f'BUY CREATE {self.dataclose[0]:2f}')
                self.order = self.buy()
                self.entry_price = self.dataclose[0]

        # Signal to CLOSE trades
        else:
            # Stop loss signal
            stop_price = self.entry_price * (1-0.03)
            if (self.dataclose[0] <= stop_price):
                self.log(f'STOP LOSS CREATE {self.dataclose[0]:2f}')
                self.order = self.close()
            # Otherwise check for swing signals and relevant filters
            elif (self.crossover < 0 or
                    self.rsi[0] > self.params.overbought or
                    self.dataclose[0] >= self.bbands.lines.top[0]):
                self.log(f'CLOSE CREATE {self.dataclose[0]:2f}')
                self.order = self.close()


# Position Sizing
class RiskSizer(bt.Sizer):
    params = (
        ('risk_per_trade', 0.01),
        ('stop_loss_pct', 0.03)
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        risk_amt = cash * self.params.risk_per_trade
        stop_loss_amt = data.close[0] * self.params.stop_loss_pct
        return int(risk_amt/stop_loss_amt)


# Backtesting strategy
if __name__ == '__main__':

    # Load historical data for backtesting
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2021, 1, 1)

    yfData = yf.download("BOX",
                         start=start_date, end=end_date, interval='1d')

    # Flatten data for backtesting with backtrader
    if isinstance(yfData.columns, pd.MultiIndex):
        yfData.columns = yfData.columns.get_level_values(0)

    data = bt.feeds.PandasData(dataname=yfData)

    # Create Cerebro Object
    cerebro = bt.Cerebro()

    # Add our Dataset
    cerebro.adddata(data)

    # Add our Swing Trade Algorithm
    cerebro.addstrategy(SwingTrade)

    # Custom Position Sizing
    cerebro.addsizer(RiskSizer)

    # Add Analyzer for Evaluation Metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe_ratio',
                        timeframe=bt.TimeFrame.Days)

    # Starting Portfolio Value
    cerebro.broker.setcash(10000)

    start_portfolio_value = cerebro.broker.get_value()

    # Run the backtest
    results = cerebro.run()

    end_portfolio_value = cerebro.broker.get_value()
    pnl = end_portfolio_value - start_portfolio_value

    # Print Relevant Results/Metrics
    print(f'Starting Portfolio Value: {start_portfolio_value:2f}')
    print(f'Final Portfolio Value: {end_portfolio_value:2f}')
    print(f'PnL: {pnl:.2f}')
    print("Sharpe Ratio (Annualized):",
          results[0].analyzers.sharpe_ratio.get_analysis()['sharperatio'])

    # Plot Results
    cerebro.plot()
    plt.show()