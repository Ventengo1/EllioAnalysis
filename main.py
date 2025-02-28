from IPython import get_ipython
from IPython.display import display
# %%
from IPython import get_ipython
from IPython.display import display
# %%

# %%

from IPython import get_ipython
import numpy as np
import pandas as pd
import math
import yfinance as yf  # Added yfinance

from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
from pylab import rcParams
# from pandas_datareader import data  # Commented out data from pandas_datareader
from scipy import signal

import os
import datetime as dt
import seaborn as sns


# %%
# PLOTTING SETUP
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 22, 10

# %%
# COMMODITY
dateTimeObj = dt.datetime.now()
today = dateTimeObj.strftime("%Y-%m-%d")

symbol = "AAPL"
date = today
# filename = '/data/%s/Yahoo_BTCUSD_d.csv.ta.csv' % symbol

# %%

# **************************************************************************
# download from yahoo the daily charts
# **************************************************************************


def download(symbol, date, days=365):
    if date is None:
        dateTimeObj = dt.datetime.now()
    else:
        dateTimeObj = dt.datetime.strptime(date, "%Y-%m-%d")

    date = dateTimeObj.strftime("%Y-%m-%d")
    date_start = (dateTimeObj - dt.timedelta(days=days)).strftime("%Y-%m-%d")

    df_source = yf.download(symbol, start=date_start, end=date)  # Use yfinance

    # Rename columns
    df_source = df_source.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "Close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    })

    df_source['Date'] = pd.to_datetime(df_source.index)
    df_source = df_source.reset_index(drop=True)
    if 'adj_close' in df_source.columns:  # Change to 'adj_close'
        df_source = df_source.drop(columns=['adj_close'])

    return df_source


# %%
def minmaxTwoMeasures(df, measureMin, measureMax, column, order=2):

    df['DateTmp'] = df.index
    x = np.array(df["DateTmp"].values)
    y1 = np.array(df[measureMin].values)
    y2 = np.array(df[measureMax].values)

    sortId = np.argsort(x)
    x = x[sortId]
    y1 = y1[sortId]
    y2 = y2[sortId]

    df[column] = 0

    maxm = signal.argrelextrema(y2, np.greater, order=order)
    minm = signal.argrelextrema(y1, np.less, order=order)
    for elem in maxm[0]:
        df.iloc[elem, df.columns.get_loc(column)] = 1
    for elem in minm[0]:
        df.iloc[elem, df.columns.get_loc(column)] = -1
    return df.drop(columns=['DateTmp'])


def isMin(df, i):
    return df["FlowMinMax"].iat[i] == -1


def isMax(df, i):
    return df["FlowMinMax"].iat[i] == 1


def distance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


def isElliottWave(df, value, i0, i1, i2, i3, i4, i5, ia, ib, ic):
    result = None

    # Check if indices are within DataFrame bounds
    indices = [i0, i1, i2, i3, i4, i5, ia, ib, ic]
    if any(i >= len(df) for i in indices):  # Check if i is within dataFrame bounds
        return None

    #Check if i1,i2,i3,i4,i5 are integers
    if not all(isinstance(i, int) for i in [i1, i2, i3, i4, i5]):
        print("Error: i1, i2, i3, i4, i5 must be integers.")
        return None

    # Get the numeric column index
    value_col_index = df.columns.get_loc(value)

    # Use .iloc[] instead of .iat[] for indexing
    isi5TheTop = (df.iloc[i5, value_col_index] > df.iloc[i1, value_col_index]) & \
                  (df.iloc[i5, value_col_index] > df.iloc[i2, value_col_index]) & \
                  (df.iloc[i5, value_col_index] > df.iloc[i3, value_col_index]) & \
                  (df.iloc[i5, value_col_index] > df.iloc[i4, value_col_index])





    if not isMin(df, i0) or not isMin(df, i2) or not isMin(
            df, i4) or not isMin(df, ia) or not isMin(df, ic):
        return result

    if not isMax(df, i1) or not isMax(df, i3) or not isMax(
            df, i5) or not isMax(df, ib):
        return result

    # Correct the comparison using bitwise operators and .iloc[0] if necessary
    isi5TheTop = (df[value].iloc[i5] > df[value].iloc[i1]) & (df[value].iloc[i5] > df[value].iloc[i2]) & (df[value].iloc[i5] > df[value].iloc[i3]) & (df[value].iloc[i5] > df[value].iloc[i4])

    if not isi5TheTop.any():
        return result

    #Example of using .iloc[0] to access a single element for float conversion:
    if not (df[value].iloc[i1] > df[value].iloc[i0]).any(): #Applying the fix to address the warning.
        return result

    if not (df[value].iloc[i1] > df[value].iloc[i2]).any():
        return result

    if not (df[value].iloc[i2] > df[value].iloc[i0]).any():
        return result

    if not (df[value].iloc[i3] > df[value].iloc[i2]).any():
        return result

    w1Len = distance(i0, float(df[value].iloc[i0].iloc[0]
                                 if isinstance(df[value].iloc[i0], pd.Series)
                                 else df[value].iloc[i0]),
                     i1,
                     float(df[value].iloc[i1].iloc[0]
                           if isinstance(df[value].iloc[i1], pd.Series)
                           else df[value].iloc[i1]))  #Example modification

    if not (df[value].iloc[i2] > df[value].iloc[i0]).any():
        return result

    if not (df[value].iloc[i3] > df[value].iloc[i4]).any():
        return result

    if not (df[value].iloc[i4] > df[value].iloc[i2]).any():
        return result
    w3Len = distance(i2, float(df[value].iloc[i2].iloc[0]
                                 if isinstance(df[value].iloc[i2], pd.Series)
                                 else df[value].iloc[i2]),
                     i3,
                     float(df[value].iloc[i3].iloc[0]
                           if isinstance(df[value].iloc[i3], pd.Series)
                           else df[value].iloc[i3]))

    if not (df[value].iloc[i4] > df[value].iloc[i1]).any():
        return result
    if not (df[value].iloc[i5] > df[value].iloc[i4]).any():
        return result

    if not (df[value].iloc[i5] > df[value].iloc[i3]).any():
        return result
    w5Len = distance(i4, float(df[value].iloc[i4].iloc[0]
                                 if isinstance(df[value].iloc[i4], pd.Series)
                                 else df[value].iloc[i4]),
                     i5,
                     float(df[value].iloc[i5].iloc[0]
                           if isinstance(df[value].iloc[i5], pd.Series)
                           else df[value].iloc[i5]))

    if (w3Len < w1Len and w3Len < w5Len):
        return result

    result = [i0, i1, i2, i3, i4, i5]

    # Correct the comparison using bitwise operators
    isi5TheTop = (df[value].iloc[i5] > df[value].iloc[ia]) & (df[value].iloc[i5] > df[value].iloc[ib]) & (df[value].iloc[i5] > df[value].iloc[ic])

    if not isi5TheTop.any():
        return result

    if not (df[value].iloc[i5] > df[value].iloc[ia]).any():
        return result

    if not (df[value].iloc[i5] > df[value].iloc[ib]).any():
        return result

    if not (df[value].iloc[ib] > df[value].iloc[ia]).any():
        return result

    if not (df[value].iloc[ia] > df[value].iloc[ic]).any():
        return result

    if not (df[value].iloc[ib] > df[value].iloc[ic]).any():
        return result

    result = [i0, i1, i2, i3, i4, i5, ia, ib, ic]

    return result


def ElliottWaveDiscovery(df, measure):

    def minRange(df, start, end):

        def localFilter(i):
            return isMin(df, i)

        return filter(localFilter, list(range(start, end)))

    def maxRange(df, start, end):

        def localFilter(i):
            return isMax(df, i)

        return filter(localFilter, list(range(start, end)))

    waves = []
    counter = 0  # Initialize counter
    for i0 in minRange(df, 0, len(df)):
        for i1 in maxRange(df, i0 + 1, len(df)):
            for i2 in minRange(df, i1 + 1, len(df)):
                for i3 in maxRange(df, i2 + 1, len(df)):
                    for i4 in minRange(df, i3 + 1, len(df)):
                        for i5 in maxRange(df, i4 + 1, len(df)):

                            isi5TheTop = (df[measure].iloc[i5] > df[measure].iloc[i1]) & (
                                df[measure].iloc[i5] > df[measure].iloc[i2]) & (
                                    df[measure].iloc[i5] > df[measure].iloc[i3]) & (
                                        df[measure].iloc[i5] > df[measure].iloc[i4])
                            if isi5TheTop.any():

                                for ia in minRange(df, i5 + 1, len(df)):
                                    for ib in maxRange(df, ia + 1, len(df)):
                                        for ic in minRange(df, ib + 1, len(df)):
                                            wave = isElliottWave(
                                                df, measure, i0, i1, i2, i3, i4,
                                                i5, ia, ib, ic)
                                            if wave is None:
                                                continue
                                            if not wave in waves:
                                                waves.append(wave)
                                                print(wave)

                            counter += 1  # Increment counter
                            if counter % 50 == 0:  # Check if counter is divisible by 100
                                print(f"Processed {counter} rows")

    return waves


# %%

def draw_wave(df, df_waves, w):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(df['Close'],
            label='Close',
            color="blue",
            linestyle="-",
            alpha=0.5)
    ax.plot(df_waves['Close'],
            label='Close',
            color="black",
            linestyle="-",
            alpha=0.5)

    ax.plot(df_waves['Close'], 'ko', markevery=None)

    df_waves.loc[:, "wave"] = None #changed to df.loc
    for i in range(0, len(w)):
       df_waves['wave'].iloc[w[i]] = df_waves['Close'].iloc[w[i]]

    df_filtered_waves = df_waves.loc[pd.notnull(df_waves.wave)]
    ax.plot(df_filtered_waves['wave'], color="red", linewidth=3.0)
    plt.show()


# %%

# select the waves the best fit the chart
def filterWaveSet(waves, min_len=6, max_len=6, extremes=True):

    result = []
    for w in waves:
        l = len(w)
        if min_len <= l and l <= max_len:
            result.append(w)

    if not extremes:
        return result

    max = 0
    for w in result:
        if w[len(w) - 1] >= max:
            max = w[len(w) - 1]

    result2 = []
    for w in result:
        if w[len(w) - 1] == max:
            result2.append(w)

    min = max
    for w in result2:
        if w[0] <= min:
            min = w[0]

    result = []
    for w in result2:
        if w[0] == min:
            result.append(w)

    return result


# %%
import math


def line(wa, wb, x):
    x1 = wa[0]
    y1 = wa[1]
    x2 = wb[0]
    y2 = wb[1]
    y = ((y2 - y1) / (x2 - x1)) * (x - x1) + y1
    return y


def elliottWaveLinearRegressionError(df_waves, w, value):
    diffquad = 0
    for i in range(1, len(w)):
        wa = [w[i - 1], df_waves[value].iloc[w[i - 1], df_waves.columns.get_loc(value)]]  # Modified line using iloc instead  # Print values #col error
        wb = [w[i], df_waves[value].iloc[w[i], df_waves.columns.get_loc(value)]]  # Modified line using iloc instead  # Print values

        for xindex in range(wa[0], wb[0]):
            yindex = df_waves[value].iloc[xindex, df_waves.columns.get_loc(value)]  # Using iloc instead
            yline = line(wa, wb, xindex)

            diffquad += (yindex - yline)**2

    return math.sqrt(diffquad) / (w[len(w) - 1] - w[0])


def findBestFitWave(df, value, waves):

    avg = np.Inf
    df_waves = df[[value, "FlowMinMax"]]
    result = []
    for w in waves:
        tmp = elliottWaveLinearRegressionError(df_waves, w, value)

        if tmp < avg:
            print(w, tmp)
            avg = tmp
            result = w
    return result


# %%

def buildWaveChainSet(waves, startwith=9):

    def addList(list, wavelist):
        k = 0
        for w in wavelist:
            k += len(w)
        key = str(k)
        if not key in list:
            list[key] = []
        list[key].append(wavelist)
        print(wavelist)
        return list

    print("chainsets")
    list = {}
    for w1 in [wave for wave in waves if len(wave) == startwith]:
        wavelist = [w1]
        if len(w1) == 9:
            for w2 in waves:
                if (len(w2) <= len(w1)):
                    if w1[len(w1) - 1] == w2[0]:
                        wavelist.append(w2)
                        addList(list, wavelist.copy())
                        wavelist.pop(-1)
        else:
            addList(list, wavelist)
    return list


# %%

# %%
# -------------------------------------------------
#  given a timeline, we generate all the possible waves
# -------------------------------------------------

if date is None:
    date = dt.datetime.now().strftime("%Y-%m-%d")
df_source = download(symbol, date, 365 * 4)

df_source["Date"] = pd.to_datetime(df_source["Date"]) #removed infer...
df_source.set_index("Date")


# %%
def ElliottWaveFindPattern(df_source, measure, granularity, dateStart,
                           dateEnd, today, extremes=True):

    mask = (dateStart <= df_source.Date) & (df_source.Date <= dateEnd)
    df = df_source.loc[mask]
    df.set_index("Date")

    FlowMinMax = minmaxTwoMeasures(df, "Close", "Close", "FlowMinMax",
                                   granularity)

    df = FlowMinMax
    df_samples = df.loc[df['FlowMinMax'] != 0]

    draw_wave(df, df_samples, [])

    print("start ", len(FlowMinMax))
    waves = ElliottWaveDiscovery(df_samples[["Close", "FlowMinMax"]], "Close")
    print("waves")
    print(waves)
    filtered_waves = filterWaveSet(waves, 5, 9, extremes=extremes)
    print("selected waves")
    print(filtered_waves)
    waves_for_len = {}
    for w in filtered_waves:
        if len(w) not in waves_for_len:
            waves_for_len[len(w)] = []
        waves_for_len[len(w)].append(w)

    for k, v in waves_for_len.items(
    ):  # Use .items() to get both keys and values
        result = findBestFitWave(df_samples, "Close",
                                 v)  # Note: v here is the value
        print("best fit")
        print(result)
        draw_wave(df, df_samples, result)

        # Add Buy/Sell/Hold Logic (Simplified Example)
        if len(result) >= 9:  # Assuming 9 points represent a complete pattern
            # Check the trend of the last wave (wave 5)
            if df_samples["Close"].iat[result[
                    4]] < df_samples["Close"].iat[result[5]]:  # Uptrend
                print("Signal: Buy")
            elif df_samples["Close"].iat[result[
                    4]] > df_samples["Close"].iat[result[5]]:  # Downtrend
                print("Signal: Sell")
            else:
                print("Signal: Hold")
        else:
            print("Signal: Hold (Incomplete Pattern)")


ElliottWaveFindPattern(df_source, "Close", 50, "2019-03-01", today, today,
                           extremes=False)
