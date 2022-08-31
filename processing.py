import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import concat

def read_data():
    base_credito = pd.read_excel("data/base_hogares.xlsx")

    # Set data as index
    base_credito['fecha'] = pd.to_datetime(base_credito['fecha'], format='%Y-%m-%d').dt.date
    base_credito= base_credito.set_index('fecha')

    # Split data in Y and X
    y = base_credito.iloc[15:, 0].copy().to_frame("credito_hogares")
    x = series_to_supervised(base_credito[12:], 3)
    # x = x.iloc[:, [0:6,9:56]]
    # x = x[x.columns[0:6].append(x.columns[9:])]
    omited_x = ["prestamo_ecuador(t-3)", "credito_ecuador(t-3)",
                "prestamo_quirografario_ecuador(t-3)",
                "tasa_activa(t-2)", "credito_banco_pacifico_ecuador(t-3)",
                "simulador_de_credito_ecuador(t-2)", "tasa_pasiva(t-2)",
                "credito_quirografario_ecuador(t-2)", "prestamo_ecuador(t-1)",
                "credito_ecuador(t-1)","precio_wti(t-1)","credito_banco_pichincha_ecuador(t-1)",
                'prestamo_ecuador(t-2)','simulador_de_credito_ecuador(t-1)','simulador_de_credito_ecuador(t-3)']
                # 'credito_quirografario_ecuador(t-3)','credito_quirografario_ecuador(t-1)']
                # "prestamo_ecuador(t-2)", "credito_ecuador(t-1)", #eliminar desde aqui para volver a 2.12 mape
                # "simulador_de_credito_ecuador(t-1)","credito_banco_pichincha_ecuador(t-1)"]
    x = x.loc[:, ~x.columns.isin(omited_x)]

    # Expanding Window Statistics Features
    # window_exp = y.expanding()
    # dataframe = concat([window_exp.min(), window_exp.mean(), window_exp.max()], axis=1)
    # dataframe = series_to_supervised(dataframe, 1)
    # dataframe.columns = ['ew_min', 'ew_mean', 'ew_max']
    # y_ew = dataframe

    # Rolling Window Statistics Features (According to pacf anf acf window must be max 3
    # width = 3
    # shifted = y.shift(width - 1)
    # window = shifted.rolling(window=width)
    # dataframe = concat([window.min(), window.mean(), window.max()], axis=1)
    # dataframe.columns = ['rw_min', 'rw_mean', 'rw_max']
    # y_rw = dataframe

    # Concatenate all the Y based features
    # y = concat([y, y_ew, y_rw], axis=1).iloc[3:, :]

    # Date Features
    x['month'] = [base_credito.index[i].month for i in range(len(x))]
    x['year'] = [base_credito.index[i].year for i in range(len(x))]
    # x = pd.get_dummies(x, columns=["year", 'month'])

    return pd.concat([y, x], axis=1)

class DataProcessing:

    def __init__(self):
        self.df = read_data()

def series_to_supervised(data, n_in=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (df.columns[j], i)) for j in range(n_vars)]
    # put it all together
    cols.append(df.prestamo_ecuador.shift(5))
    names.append("prestamo_ecuador(t-5)")
    cols.append(df.prestamo_ecuador.shift(6))
    names.append("prestamo_ecuador(t-6)")
    cols.append(df.credito_ecuador.shift(6))
    names.append("credito_ecuador(t-6)")
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if data.shape[1] > 15:
        agg.dropna(subset=['credito_hogares(t-3)'],inplace=True)
    return agg
