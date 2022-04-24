import pandas as pd, numpy as np
pd.options.mode.chained_assignment = None
import yfinance as yf

N = 14 # Number of Periods for Indicator Parameters
N1 = 8 # Fast Period for MACD
## Nifty 50
tickers = 'AXISBANK.NS ADANIPORTS.NS ASIANPAINT.NS BAJAJ-AUTO.NS BAJFINANCE.NS BAJAJFINSV.NS BPCL.NS BHARTIARTL.NS BRITANNIA.NS CIPLA.NS DIVISLAB.NS DRREDDY.NS EICHERMOT.NS GRASIM.NS HCLTECH.NS HDFC.NS HDFCBANK.NS HEROMOTOCO.NS HINDALCO.NS HINDUNILVR.NS ICICIBANK.NS ITC.NS IOC.NS INDUSINDBK.NS INFY.NS JSWSTEEL.NS KOTAKBANK.NS LT.NS M&M.NS MARUTI.NS NTPC.NS NESTLEIND.NS ONGC.NS POWERGRID.NS RELIANCE.NS SBIN.NS SHREECEM.NS SUNPHARMA.NS TCS.NS TATACONSUM.NS TATAMOTORS.NS TATASTEEL.NS TECHM.NS TITAN.NS UPL.NS ULTRACEMCO.NS WIPRO.NS'
index_ticker = '^NSEI'
start_date = '2010-01-01'
end_date = '2021-12-31'
duration = 12

# Use with period2 > period1 ONLY
def MACD(data: pd.DataFrame, period1: int, period2: int):
    df = data.copy()
    df['EMA1'] = df['Close'].ewm(alpha = 1 / period1, adjust = True).mean()
    df['EMA2'] = df['Close'].ewm(alpha = 1 / period2, adjust = True).mean()
    df['MACD'] = df['EMA1'] - df['EMA2']
    return df

def VWADX(data: pd.DataFrame, period: int):
    df = data.copy()
    expwts = [0] * period
    for i in range(0, period):
        expwts[i] = (1 - 1 / period) ** (period - i - 1)
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    del df['H-L'], df['H-C'], df['L-C']
    df['AVWTR'] = df['TR']
    for i in range(period - 1, len(df['TR'])):
        try:
            df['AVWTR'][i] = np.average(df['TR'][i - period + 1: i + 1], weights = np.multiply(df['Volume'][i - period + 1: i + 1], expwts[:]))
        except:
            df['AVWTR'][i] = 0;
    df['H-pH'] = df['High'] - df['High'].shift(1)
    df['pL-L'] = df['Low'].shift(1) - df['Low']
    df['+DX'] = np.where(
        (df['H-pH'] > df['pL-L']) & (df['H-pH'] > 0),
        df['H-pH'],
        0.0
    )
    df['-DX'] = np.where(
        (df['H-pH'] < df['pL-L']) & (df['pL-L'] > 0),
        df['pL-L'],
        0.0
    )
    del df['H-pH'], df['pL-L']
    df['VWS+DM'] = df['+DX'].ewm(alpha = 1 / period, adjust = True).mean()
    df['VWS-DM'] = df['-DX'].ewm(alpha = 1 / period, adjust = True).mean()
    for i in range(period - 1, len(df['+DX'])):
        try:
            df['VWS+DM'][i] = np.average(df['+DX'][i - period + 1: i + 1], weights = np.multiply(df['Volume'][i - period + 1: i + 1], expwts[:]))
        except:
            df['VWS+DM'][i] = 0
        try:
            df['VWS-DM'][i] = np.average(df['-DX'][i - period + 1: i + 1], weights = np.multiply(df['Volume'][i - period + 1: i + 1], expwts[:]))
        except:
            df['VWS-DM'][i] = 0
    try:
        df['VW+DMI'] = (df['VWS+DM'] / df['AVWTR']) * 100
    except:
        df['VW+DMI'] = 0
    try:
        df['VW-DMI'] = (df['VWS-DM'] / df['AVWTR']) * 100
    except:
        df['VW-DMI'] = 0
    del df['VWS+DM'], df['VWS-DM']
    try:
        df['VWDX'] = (np.abs(df['VW+DMI'] - df['VW-DMI']) / (df['VW+DMI'] + df['VW-DMI'])) * 100
    except:
        df['VWDX'] = 0
    df['VWADX'] = df['VWDX'].ewm(alpha = 1 / period, adjust = True).mean()
    for i in range(period - 1, len(df['VWDX'])):
        try:
            df['VWADX'][i] = np.average(df['VWDX'][i - period + 1: i + 1], weights = np.multiply(df['Volume'][i - period + 1: i + 1], expwts[:]))
        except:
            df['VWADX'][i] = 0
    del df['AVWTR'], df['VWDX'], df['TR'], df['-DX'], df['+DX'], df['VW+DMI'], df['VW-DMI']
    return df

def ADX(data: pd.DataFrame, period: int):
    df = data.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    del df['H-L'], df['H-C'], df['L-C']
    df['ATR'] = df['TR'].ewm(alpha = 1 / period, adjust = False).mean()
    df['H-pH'] = df['High'] - df['High'].shift(1)
    df['pL-L'] = df['Low'].shift(1) - df['Low']
    df['+DX'] = np.where(
        (df['H-pH'] > df['pL-L']) & (df['H-pH']>0),
        df['H-pH'],
        0.0
    )
    df['-DX'] = np.where(
        (df['H-pH'] < df['pL-L']) & (df['pL-L']>0),
        df['pL-L'],
        0.0
    )
    del df['H-pH'], df['pL-L']
    df['S+DM'] = df['+DX'].ewm(alpha = 1 / period, adjust = False).mean()
    df['S-DM'] = df['-DX'].ewm(alpha = 1 / period, adjust = False).mean()
    df['+DMI'] = (df['S+DM'] / df['ATR']) * 100
    df['-DMI'] = (df['S-DM'] / df['ATR']) * 100
    del df['S+DM'], df['S-DM']
    df['DX'] = (np.abs(df['+DMI'] - df['-DMI'])/(df['+DMI'] + df['-DMI'])) * 100
    df['ADX'] = df['DX'].ewm(alpha = 1 / period, adjust = False).mean()
    del df['ATR'], df['DX'], df['TR'], df['-DX'], df['+DX'], df['+DMI'], df['-DMI']
    return df

def trend_macd(data: pd.DataFrame):
    df = data.copy()
    df['trend'] = np.sign(df['MACD'])
    df['trend_long'] =  np.where(
        df['trend'] == 1.0,
        1.0,
        0.0
    )
    return df

def weight_adx(data: pd.DataFrame):
    df = data.copy()
    df['wt_ADX_1'] = np.where(
        df['ADX'] > 25,
        df['ADX'],
        0.0
    )
    df['wt_ADX_2'] = np.where(
        df['ADX'] > 25,
        df['ADX'] - 25,
        0.0
    )
    df['wt_ADX_1'] *= df['trend_long']
    df['wt_ADX_2'] *= df['trend_long']
    return df

def weight_vwadx(data: pd.DataFrame):
    df = data.copy()
    df['wt_VWADX_1'] = np.where(
        df['VWADX'] > 25,
        df['VWADX'],
        0.0
    )
    df['wt_VWADX_2'] = np.where(
        df['VWADX'] > 25,
        df['VWADX'] - 25,
        0.0
    )
    df['wt_VWADX_1'] *= df['trend_long']
    df['wt_VWADX_2'] *= df['trend_long']
    return df

ticker_list = tickers.split(' ')
stock_data = []
for i in range(len(ticker_list)):
    try:
        stock_data.append(yf.download(ticker_list[i], start = start_date, end = end_date))
    except:
        print(ticker_list[i])
perf_data = []
perf_data.append(yf.download(index_ticker, start = start_date, end = end_date))

# Data Correction for HCLTECH.NS stock
for date in stock_data[14].index:
    there = False
    for date2 in stock_data[0].index:
        if date == date2:
            there = True
            break
    if not there:
        stock_data[14] = stock_data[14].drop(date)

U = len(stock_data)
T = len(stock_data[0])
L = len(perf_data[0])
for i in range(U):
    df = stock_data[i]
    df = ADX(df, N)
    df = VWADX(df, N)
    df = MACD(df, N1, N)
    df = trend_macd(df)
    df = weight_adx(df)
    df = weight_vwadx(df)
    stock_data[i] = df

vwadx_weights_1 = {'Date':[], 'Weights':[]}
vwadx_weights_2 = {'Date':[], 'Weights':[]}
adx_weights_1 = {'Date':[], 'Weights':[]}
adx_weights_2 = {'Date':[], 'Weights':[]}
vwadx_weights_1['Weights'].append([0] * U)
vwadx_weights_2['Weights'].append([0] * U)
adx_weights_1['Weights'].append([0] * U)
adx_weights_2['Weights'].append([0] * U)
for date in stock_data[0].index:
    curr_wt_vwadx_1 = []
    curr_wt_vwadx_2 = []
    curr_wt_adx_1 = []
    curr_wt_adx_2 = []
    vwadx_weights_1['Date'].append(date)
    vwadx_weights_2['Date'].append(date)
    adx_weights_1['Date'].append(date)
    adx_weights_2['Date'].append(date)
    if(date == stock_data[0].index[-1]):
        break
    for i in range(U):
        curr_wt_vwadx_1.append(stock_data[i]['wt_VWADX_1'][date])
        curr_wt_vwadx_2.append(stock_data[i]['wt_VWADX_2'][date])
        curr_wt_adx_1.append(stock_data[i]['wt_ADX_1'][date])
        curr_wt_adx_2.append(stock_data[i]['wt_ADX_2'][date])
    vwadx_weights_1['Weights'].append(curr_wt_vwadx_1)
    vwadx_weights_2['Weights'].append(curr_wt_vwadx_2)
    adx_weights_1['Weights'].append(curr_wt_adx_1)
    adx_weights_2['Weights'].append(curr_wt_adx_2)
vwadx_wts_1 = pd.DataFrame.from_dict(vwadx_weights_1)
vwadx_wts_2 = pd.DataFrame.from_dict(vwadx_weights_2)
adx_wts_1 = pd.DataFrame.from_dict(adx_weights_1)
adx_wts_2 = pd.DataFrame.from_dict(adx_weights_2)

for date in vwadx_wts_1.index:
    if sum(vwadx_wts_1['Weights'][date]) != 0:
        vwadx_wts_1['Weights'][date] = [i/sum(vwadx_wts_1['Weights'][date]) for i in vwadx_wts_1['Weights'][date]]
    if sum(vwadx_wts_2['Weights'][date]) != 0:
        vwadx_wts_2['Weights'][date] = [i/sum(vwadx_wts_2['Weights'][date]) for i in vwadx_wts_2['Weights'][date]]
    if sum(adx_wts_1['Weights'][date]) != 0:
        adx_wts_1['Weights'][date] = [i/sum(adx_wts_1['Weights'][date]) for i in adx_wts_1['Weights'][date]]
    if sum(adx_wts_2['Weights'][date]) != 0:
        adx_wts_2['Weights'][date] = [i/sum(adx_wts_2['Weights'][date]) for i in adx_wts_2['Weights'][date]]

port_perf = {'Date':[], 'VWADX_1':[], 'VWADX_2':[], 'ADX_1':[], 'ADX_2':[]}
port_perf['Date'].append(stock_data[0].index[0])
port_perf['VWADX_1'].append(1.0)
port_perf['VWADX_2'].append(1.0)
port_perf['ADX_1'].append(1.0)
port_perf['ADX_2'].append(1.0)
date1 = vwadx_wts_1.index[0]
for date in vwadx_wts_1.index[1:]:
    port_perf['Date'].append(stock_data[0].index[date])
    val_vwadx_1 = 0
    val_vwadx_2 = 0
    val_adx_1 = 0
    val_adx_2 = 0
    for i in range(U):
        val_vwadx_1 += vwadx_wts_1['Weights'][date][i] * stock_data[i]['Close'][date] / stock_data[i]['Close'][date1]
        val_vwadx_2 += vwadx_wts_2['Weights'][date][i] * stock_data[i]['Close'][date] / stock_data[i]['Close'][date1]
        val_adx_1 += adx_wts_1['Weights'][date][i] * stock_data[i]['Close'][date] / stock_data[i]['Close'][date1]
        val_adx_2 += adx_wts_2['Weights'][date][i] * stock_data[i]['Close'][date] / stock_data[i]['Close'][date1]
    val_vwadx_1 += 1.0 - sum(vwadx_wts_1['Weights'][date])
    val_vwadx_2 += 1.0 - sum(vwadx_wts_2['Weights'][date])
    val_adx_1 += 1.0 - sum(adx_wts_1['Weights'][date])
    val_adx_2 += 1.0 - sum(adx_wts_2['Weights'][date])
    port_perf['VWADX_1'].append(val_vwadx_1 * port_perf['VWADX_1'][-1])
    port_perf['VWADX_2'].append(val_vwadx_2 * port_perf['VWADX_2'][-1])
    port_perf['ADX_1'].append(val_adx_1 * port_perf['ADX_1'][-1])
    port_perf['ADX_2'].append(val_adx_2 * port_perf['ADX_2'][-1])
    date1 = date
port_perf = pd.DataFrame.from_dict(port_perf)
port_perf.set_index('Date')

port_perf.to_csv('Portfolio Performance Comparison.csv')
perf_data[0].to_csv('NIFTY 50 Performance.csv')