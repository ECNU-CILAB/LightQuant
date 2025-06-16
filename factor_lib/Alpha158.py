import pandas as pd
import numpy as np


def calc_alpha158_factors(df):
    result = pd.DataFrame(index=df.index)

    C = df['Close']
    O = df['Open']
    H = df['High']
    L = df['Low']
    V = df['Volume']
    A = df['Amount']
    R = df['Returns']

    # Factor 1: Ratio of price and volume
    result['alpha_1'] = (C - O) / (H - L) * V

    # Factor 2: Ratio of price and volume (based on amount)
    result['alpha_2'] = (C - O) / (H - L) * A

    # Factor 3: Momentum factor based on price
    result['alpha_3'] = C.diff(5) / C.shift(5)

    # Factor 4: Momentum factor based on volume
    result['alpha_4'] = V.diff(5) / V.shift(5)

    # Factor 5: 5-day moving average of daily returns
    result['alpha_5'] = R.rolling(5).mean()

    # Factor 6: Ratio of the current close price to the lowest price over the past N days
    result['alpha_6'] = (C - L.rolling(20).min()) / (H.rolling(20).max() - L.rolling(20).min())

    # Factor 7: Ratio of the current close price to the highest price over the past N days
    result['alpha_7'] = (C - H.rolling(20).max()) / (H.rolling(20).max() - L.rolling(20).min())

    # Factor 8: Relative strength of the difference between the close price and the open price
    result['alpha_8'] = (C - O) / C.rolling(10).mean()

    # Factor 9: Ratio of the close price to the 20-day moving average of close price
    result['alpha_9'] = C / C.rolling(20).mean()

    # Factor 10: Volatility between the highest and lowest prices
    result['alpha_10'] = (H - L) / O

    # Factor 11: Maximum price increase over the past 5 days
    result['alpha_11'] = (C - C.shift(5)) / C.shift(5)

    # Factor 12: Ratio of the 10-day moving average of volume to the current volume
    result['alpha_12'] = V / V.rolling(10).mean()

    # Factor 13: Ratio of close price to volume
    result['alpha_13'] = C / V

    # Factor 14: Difference between the close price and open price relative to open price
    result['alpha_14'] = (C - O) / O

    # Factor 15: Percentage change from the previous day's close price
    result['alpha_15'] = (C - C.shift(1)) / C.shift(1)

    # Factor 16: Ratio of the difference between the close and open price to the range between high and low price
    result['alpha_16'] = (C - O) / (H - L)

    # Factor 17: Volatility of the close price (standard deviation)
    result['alpha_17'] = C.rolling(20).std()

    # Factor 18: Volatility of the volume (standard deviation)
    result['alpha_18'] = V.rolling(20).std()

    # Factor 19: Ratio of the close price to the highest close price over the past N days
    result['alpha_19'] = C / C.rolling(20).max()

    # Factor 20: Ratio of the close price to the lowest close price over the past N days
    result['alpha_20'] = C / C.rolling(20).min()

    # Factor 21: 10-day momentum based on close price
    result['alpha_21'] = C.diff(10) / C.shift(10)

    # Factor 22: Ratio of the difference between the highest and lowest prices to the close price
    result['alpha_22'] = (H - L) / C

    # Factor 23: Volume momentum based on 5-day rolling average
    result['alpha_23'] = V.diff(5) / V.rolling(5).mean()

    # Factor 24: Moving average convergence divergence (MACD) signal line
    result['alpha_24'] = C.ewm(span=12).mean() - C.ewm(span=26).mean()

    # Factor 25: 10-day relative strength index (RSI)
    delta = C.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(10).mean()
    avg_loss = loss.rolling(10).mean()
    rs = avg_gain / avg_loss
    result['alpha_25'] = 100 - (100 / (1 + rs))

    # Factor 26: Difference between close and open price scaled by range (H - L)
    result['alpha_26'] = (C - O) / (H - L)

    # Factor 27: Volatility based on 20-day rolling standard deviation of close price
    result['alpha_27'] = C.rolling(20).std()

    # Factor 28: Percentage difference between high and low prices
    result['alpha_28'] = (H - L) / C

    # Factor 29: Rate of change of close price over 15 days
    result['alpha_29'] = C.diff(15) / C.shift(15)

    # Factor 30: 10-day moving average of returns
    result['alpha_30'] = R.rolling(10).mean()

    # Factor 31: Close price / 30-day moving average of close price
    result['alpha_31'] = C / C.rolling(30).mean()

    # Factor 32: Percentage of close price change relative to 30-day moving average
    result['alpha_32'] = (C - C.rolling(30).mean()) / C.rolling(30).mean()

    # Factor 33: Standard deviation of returns over the last 5 days
    result['alpha_33'] = R.rolling(5).std()

    # Factor 34: 14-day relative strength index (RSI)
    delta = C.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    result['alpha_34'] = 100 - (100 / (1 + rs))

    # Factor 35: Open-close price difference normalized by range
    result['alpha_35'] = (O - C) / (H - L)

    # Factor 36: Close price difference over 5 days, scaled by range
    result['alpha_36'] = (C.diff(5)) / (H - L)

    # Factor 37: Close price / Highest close price in the last 20 days
    result['alpha_37'] = C / C.rolling(20).max()

    # Factor 38: Close price / Lowest close price in the last 20 days
    result['alpha_38'] = C / C.rolling(20).min()

    # Factor 39: 30-day moving average of volume
    result['alpha_39'] = V.rolling(30).mean()

    # Factor 40: Bollinger bands (upper band)
    sma = C.rolling(20).mean()
    std = C.rolling(20).std()
    result['alpha_40'] = sma + (2 * std)

    # Factor 41: Bollinger bands (lower band)
    result['alpha_41'] = sma - (2 * std)

    # Factor 42: Rate of change of volume over 10 days
    result['alpha_42'] = V.diff(10) / V.shift(10)

    # Factor 43: Close price difference with 50-day moving average
    result['alpha_43'] = (C - C.rolling(50).mean()) / C.rolling(50).mean()

    # Factor 44: 10-day volatility of returns (standard deviation)
    result['alpha_44'] = R.rolling(10).std()

    # Factor 45: Momentum based on 20-day price difference
    result['alpha_45'] = C.diff(20) / C.shift(20)

    # Factor 46: Moving average of returns over 50 days
    result['alpha_46'] = R.rolling(50).mean()

    # Factor 47: 5-day exponential moving average (EMA) of returns
    result['alpha_47'] = R.ewm(span=5).mean()

    # Factor 48: Relative strength of volume, based on 10-day moving average
    result['alpha_48'] = V / V.rolling(10).mean()

    # Factor 49: Rate of change of close price over 30 days
    result['alpha_49'] = C.diff(30) / C.shift(30)

    # Factor 50: Average true range (ATR) over 14 days
    tr = pd.DataFrame({
        'tr1': H - L,
        'tr2': abs(H - C.shift(1)),
        'tr3': abs(L - C.shift(1)),
    })
    atr = tr.max(axis=1).rolling(14).mean()
    result['alpha_50'] = atr

    # Factor 51: Rolling max-min ratio of close over 20 days
    result['alpha_51'] = (C.rolling(20).max() - C.rolling(20).min()) / C.rolling(20).min()

    # Factor 52: Price rate of change over 7 days
    result['alpha_52'] = C.pct_change(7)

    # Factor 53: Volume spike compared to 10-day mean
    result['alpha_53'] = V / V.rolling(10).mean()

    # Factor 54: High price compared to 10-day high
    result['alpha_54'] = H / H.rolling(10).max()

    # Factor 55: Close price vs median price (high+low)/2
    result['alpha_55'] = (C - (H + L) / 2) / C

    # Factor 56: True range normalized by close
    tr = pd.concat([
        H - L,
        abs(H - C.shift(1)),
        abs(L - C.shift(1))
    ], axis=1).max(axis=1)
    result['alpha_56'] = tr / C

    # Factor 57: Z-score of close over 10-day window
    result['alpha_57'] = (C - C.rolling(10).mean()) / C.rolling(10).std()

    # Factor 58: Skewness of returns over 15-day window
    result['alpha_58'] = R.rolling(15).skew()

    # Factor 59: Kurtosis of returns over 15-day window
    result['alpha_59'] = R.rolling(15).kurt()

    # Factor 60: Price gap between today's open and yesterday's close
    result['alpha_60'] = (O - C.shift(1)) / C.shift(1)

    # Factor 61: Momentum of close to open over 3 days
    result['alpha_61'] = ((C - O) / O).rolling(3).mean()

    # Factor 62: Difference between today's high and yesterday's high
    result['alpha_62'] = H - H.shift(1)

    # Factor 63: Difference between today's low and yesterday's low
    result['alpha_63'] = L - L.shift(1)

    # Factor 64: Log return over 5 days
    result['alpha_64'] = np.log(C / C.shift(5))

    # Factor 65: Rolling correlation between close and volume (10 days)
    result['alpha_65'] = C.rolling(10).corr(V)

    # Factor 66: Rolling correlation between high and volume (10 days)
    result['alpha_66'] = H.rolling(10).corr(V)

    # Factor 67: Rolling correlation between low and volume (10 days)
    result['alpha_67'] = L.rolling(10).corr(V)

    # Factor 68: Rolling beta of close price against market index (assumes df['Market'] exists)
    if 'Market' in df.columns:
        cov = C.rolling(20).cov(df['Market'])
        var = df['Market'].rolling(20).var()
        result['alpha_68'] = cov / var
    else:
        result['alpha_68'] = np.nan

    # Factor 69: Cumulative return over past 10 days
    result['alpha_69'] = (1 + R).rolling(10).apply(np.prod, raw=True) - 1

    # Factor 70: Close relative to open over 10-day average
    result['alpha_70'] = ((C - O) / O).rolling(10).mean()

    # Factor 71: Rolling min of low prices
    result['alpha_71'] = L.rolling(15).min()

    # Factor 72: Rolling max of high prices
    result['alpha_72'] = H.rolling(15).max()

    # Factor 73: Range (high - low) volatility
    result['alpha_73'] = (H - L).rolling(10).std()

    # Factor 74: Mean reversion indicator (close - MA)
    result['alpha_74'] = C - C.rolling(20).mean()

    # Factor 75: Bollinger Band width
    ma = C.rolling(20).mean()
    std = C.rolling(20).std()
    result['alpha_75'] = (ma + 2 * std - (ma - 2 * std)) / ma

    # Factor 76: Rolling average of high-low difference
    result['alpha_76'] = (H - L).rolling(10).mean()

    # Factor 77: EMA of close (span=10)
    result['alpha_77'] = C.ewm(span=10).mean()

    # Factor 78: Price breakout indicator
    result['alpha_78'] = (C > C.rolling(20).max()).astype(int)

    # Factor 79: Price breakdown indicator
    result['alpha_79'] = (C < C.rolling(20).min()).astype(int)

    # Factor 80: Volatility breakout (high - low > 2x 20-day average)
    result['alpha_80'] = ((H - L) > 2 * (H - L).rolling(20).mean()).astype(int)

    # Factor 81: 20-day rolling mean of close price
    result['alpha_81'] = C.rolling(20).mean()

    # Factor 82: 20-day rolling standard deviation of close price
    result['alpha_82'] = C.rolling(20).std()

    # Factor 83: Close price difference normalized by the 50-day moving average
    result['alpha_83'] = (C - C.rolling(50).mean()) / C.rolling(50).mean()

    # Factor 84: Close price relative to the maximum close price over the last 50 days
    result['alpha_84'] = C / C.rolling(50).max()

    # Factor 85: Close price relative to the minimum close price over the last 50 days
    result['alpha_85'] = C / C.rolling(50).min()

    # Factor 86: High minus low price divided by the range over 20 days
    result['alpha_86'] = (H - L) / (H.rolling(20).max() - L.rolling(20).min())

    # Factor 87: Difference between close and open price, scaled by the volatility
    result['alpha_87'] = (C - O) / C.rolling(20).std()

    # Factor 88: Relative volatility of close price compared to a rolling window of 30 days
    result['alpha_88'] = C.rolling(30).std() / C.rolling(30).mean()

    # Factor 89: Cumulative sum of daily returns over 30 days
    result['alpha_89'] = (1 + R).rolling(30).apply(np.prod, raw=True) - 1

    # Factor 90: Rolling beta of close price against a market index (assumes df['Market'] exists)
    if 'Market' in df.columns:
        cov = C.rolling(20).cov(df['Market'])
        var = df['Market'].rolling(20).var()
        result['alpha_90'] = cov / var
    else:
        result['alpha_90'] = np.nan

    # Factor 91: Rate of change of volume over 30 days
    result['alpha_91'] = V.diff(30) / V.shift(30)

    # Factor 92: Price movement relative to a 5-day moving average of high prices
    result['alpha_92'] = (C - H.rolling(5).mean()) / C

    # Factor 93: Price rate of change over 60 days
    result['alpha_93'] = C.pct_change(60)

    # Factor 94: Close price compared to the 50-day exponential moving average (EMA)
    result['alpha_94'] = C / C.ewm(span=50).mean()

    # Factor 95: Percentage change between close and the 20-day simple moving average (SMA)
    result['alpha_95'] = (C - C.rolling(20).mean()) / C.rolling(20).mean()

    # Factor 96: Weighted moving average (WMA) of the close price over 20 days
    weights = np.arange(1, 21)
    result['alpha_96'] = C.rolling(20).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    # Factor 97: Logarithmic rate of change (log return) of close price over 60 days
    result['alpha_97'] = np.log(C / C.shift(60))

    # Factor 98: Percentage change of high price over 20 days
    result['alpha_98'] = H.pct_change(20)

    # Factor 99: Percentage change of low price over 20 days
    result['alpha_99'] = L.pct_change(20)

    # Factor 100: Rolling 20-day correlation between close and volume
    result['alpha_100'] = C.rolling(20).corr(V)

    # Factor 101: Price difference between open and close divided by the average of high and low
    result['alpha_101'] = (C - O) / ((H + L) / 2)

    # Factor 102: 5-day exponential moving average of returns
    result['alpha_102'] = R.ewm(span=5).mean()

    # Factor 103: Momentum indicator of close prices over the last 3 days
    result['alpha_103'] = (C.diff(3)) / C.shift(3)

    # Factor 104: Volume change relative to the 20-day moving average of volume
    result['alpha_104'] = V / V.rolling(20).mean()

    # Factor 105: Relative strength index (RSI) over 14 days
    delta = C.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    result['alpha_105'] = 100 - (100 / (1 + rs))

    # Factor 106: 50-day moving average of the close price
    result['alpha_106'] = C.rolling(50).mean()

    # Factor 107: 50-day rolling standard deviation of close price
    result['alpha_107'] = C.rolling(50).std()

    # Factor 108: Rate of change of close price relative to 20-day moving average
    result['alpha_108'] = (C - C.rolling(20).mean()) / C.rolling(20).mean()

    # Factor 109: Price change over the past 3 days divided by volume
    result['alpha_109'] = C.diff(3) / V

    # Factor 110: Rate of change of the difference between the high and low prices over 20 days
    result['alpha_110'] = (H - L).pct_change(20)

    # Factor 111: Rolling 30-day correlation between close price and volume
    result['alpha_111'] = C.rolling(30).corr(V)

    # Factor 112: Close price compared to the median of the high and low prices over 14 days
    result['alpha_112'] = (C - (H + L) / 2) / C

    # Factor 113: Moving average of price differences over the last 20 days
    result['alpha_113'] = (C.diff(20)).rolling(20).mean()

    # Factor 114: Difference between high and low prices, normalized by close price
    result['alpha_114'] = (H - L) / C

    # Factor 115: 10-day exponential moving average of volume
    result['alpha_115'] = V.ewm(span=10).mean()

    # Factor 116: 10-day standard deviation of returns
    result['alpha_116'] = R.rolling(10).std()

    # Factor 117: Price change over the last 3 days, relative to its 20-day moving average
    result['alpha_117'] = (C.diff(3)) / C.rolling(20).mean()

    # Factor 118: Close price rate of change over 15 days
    result['alpha_118'] = C.pct_change(15)

    # Factor 119: Relative volume change compared to a 30-day rolling average
    result['alpha_119'] = V / V.rolling(30).mean()

    # Factor 120: Close price normalized by the 30-day rolling high
    result['alpha_120'] = C / C.rolling(30).max()

    # Factor 121: Close price normalized by the 30-day rolling low
    result['alpha_121'] = C / C.rolling(30).min()

    # Factor 122: Moving average of close price differences over 10 days
    result['alpha_122'] = (C.diff(10)).rolling(10).mean()

    # Factor 123: Percentage change of high price relative to its 30-day moving average
    result['alpha_123'] = (H - H.rolling(30).mean()) / H.rolling(30).mean()

    # Factor 124: Percentage change of low price relative to its 30-day moving average
    result['alpha_124'] = (L - L.rolling(30).mean()) / L.rolling(30).mean()

    # Factor 125: 5-day standard deviation of close prices
    result['alpha_125'] = C.rolling(5).std()

    # Factor 126: 5-day moving average of returns
    result['alpha_126'] = R.rolling(5).mean()

    # Factor 127: 5-day price momentum (percentage change)
    result['alpha_127'] = C.pct_change(5)

    # Factor 128: 5-day volume momentum (percentage change)
    result['alpha_128'] = V.pct_change(5)

    # Factor 129: Price difference between open and close relative to the range (high-low)
    result['alpha_129'] = (C - O) / (H - L)

    # Factor 130: Average true range (ATR) over 10 days
    tr = pd.concat([
        H - L,
        abs(H - C.shift(1)),
        abs(L - C.shift(1))
    ], axis=1).max(axis=1)
    result['alpha_130'] = tr.rolling(10).mean()

    # Factor 131: Percentage change between close and the 60-day moving average
    result['alpha_131'] = (C - C.rolling(60).mean()) / C.rolling(60).mean()

    # Factor 132: 20-day momentum of price difference between high and low
    result['alpha_132'] = (H - L).diff(20) / (H - L).shift(20)

    # Factor 133: Moving average of the rate of change of volume over 10 days
    result['alpha_133'] = V.pct_change(10).rolling(10).mean()

    # Factor 134: 20-day moving average of price change
    result['alpha_134'] = C.diff(20).rolling(20).mean()

    # Factor 135: Rolling sum of the absolute daily returns over 20 days
    result['alpha_135'] = R.abs().rolling(20).sum()

    # Factor 136: Difference between close price and the 20-day moving average of close price
    result['alpha_136'] = C - C.rolling(20).mean()

    # Factor 137: Cumulative sum of returns over 20 days
    result['alpha_137'] = (1 + R).rolling(20).apply(np.prod, raw=True) - 1

    # Factor 138: Difference between close and open price normalized by the 20-day average of the high-low range
    result['alpha_138'] = (C - O) / ((H - L).rolling(20).mean())

    # Factor 139: Moving average of close price over the last 100 days
    result['alpha_139'] = C.rolling(100).mean()

    # Factor 140: 100-day standard deviation of the close price
    result['alpha_140'] = C.rolling(100).std()

    # Factor 141: 10-day moving average of volume
    result['alpha_141'] = V.rolling(10).mean()

    # Factor 142: 10-day percentage change in volume
    result['alpha_142'] = V.pct_change(10)

    # Factor 143: 30-day correlation between close price and volume
    result['alpha_143'] = C.rolling(30).corr(V)

    # Factor 144: 15-day exponential moving average of close price
    result['alpha_144'] = C.ewm(span=15).mean()

    # Factor 145: 15-day percentage change in close price
    result['alpha_145'] = C.pct_change(15)

    # Factor 146: 30-day rolling skewness of close price
    result['alpha_146'] = C.rolling(30).skew()

    # Factor 147: 30-day rolling kurtosis of close price
    result['alpha_147'] = C.rolling(30).kurt()

    # Factor 148: 30-day rolling correlation between high and low prices
    result['alpha_148'] = H.rolling(30).corr(L)

    # Factor 149: Relative strength index (RSI) over 14 days
    delta = C.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    result['alpha_149'] = 100 - (100 / (1 + rs))

    # Factor 150: 14-day simple moving average of the relative strength index (RSI)
    result['alpha_150'] = result['alpha_149'].rolling(14).mean()

    # Factor 151: Percentage change of close price relative to its 100-day moving average
    result['alpha_151'] = (C - C.rolling(100).mean()) / C.rolling(100).mean()

    # Factor 152: Difference between open and close price relative to 5-day average true range
    tr = pd.concat([H - L, abs(H - C.shift(1)), abs(L - C.shift(1))], axis=1).max(axis=1)
    result['alpha_152'] = (O - C) / tr.rolling(5).mean()

    # Factor 153: 20-day rolling correlation between close price and open price
    result['alpha_153'] = C.rolling(20).corr(O)

    # Factor 154: Close price normalized by the 5-day moving average of high price
    result['alpha_154'] = C / H.rolling(5).mean()

    # Factor 155: Close price normalized by the 5-day moving average of low price
    result['alpha_155'] = C / L.rolling(5).mean()

    # Factor 156: 50-day exponential moving average of returns
    result['alpha_156'] = R.ewm(span=50).mean()

    # Factor 157: 50-day moving average of the absolute daily returns
    result['alpha_157'] = R.abs().rolling(50).mean()

    # Factor 158: 60-day standard deviation of the close price
    result['alpha_158'] = C.rolling(60).std()

    return result

