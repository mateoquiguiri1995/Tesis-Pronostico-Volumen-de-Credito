import random
import arch
import pandas as pd  # load and manipulate data and hot encoding
import numpy as np  # calculate the mean and standard deviation
import xgboost as xgb  # Xgboost stuff
import statsmodels.api as sm
import statsmodels.stats as sms
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from xgboost import plot_importance
import scipy as sp
from processing import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller


def cycle_ts_train_test_split(df, n_test=69):
    df = df[:180]
    return df.iloc[:-n_test, 1:], df.iloc[-n_test:, 1:], df.iloc[:-n_test, 0], df.iloc[-n_test:, 0]


def plot_predictions(y_test, predictions, cycle):
    """
    Plot the predicted and observed points given cycle
    Arguments:
        y_test: Sequence of observed values as a series.
        predictions: Sequence of predicted values as a series.
        cycle: Boolean whether or not plot considering economic cycle.
    Returns:
        Plot.
    """
    plt.figure(figsize=(11, 8))
    plt.plot(y_test, marker="o", label='Observado', linestyle="dashed", color="black")
    plt.plot(predictions, marker="o", label='Pronosticado', color="dodgerblue")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    if cycle:
        plt.axvspan("2015-08-31", "2017-03-31", label="Recesión BCE", color='silver', alpha=0.6)
        plt.axvspan("2020-03-31", "2021-02-28", color='silver', alpha=0.6)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=5))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Millones de Dólares ($)", fontsize=12)
    plt.legend(fontsize=14, loc=2)
    plt.gcf().autofmt_xdate(bottom=0.2, rotation=45)
    plt.show()


class BaselinePredictor(DataProcessing):
    def __init__(self, cycle=False):
        DataProcessing.__init__(self)
        self.x_train, self.x_test, self.y_train, self.y_test = self.ts_train_test_split()
        self.cycle = cycle
        self.mape, self.rmse, self.predictions = self.model()

    def ts_train_test_split(self, n_test=12):
        df = self.df
        return df.iloc[:-n_test, 1:], df.iloc[-n_test:, 1:], df.iloc[:-n_test, 0], df.iloc[-n_test:, 0]

    def model(self):
        if self.cycle:
            self.x_train, self.x_test, self.y_train, self.y_test = cycle_ts_train_test_split(self.df, n_test=69)
        history = list(self.y_train.values)
        predictions = list()
        for t in range(len(self.y_test)):
            output = history[-1]
            predictions.append(output)
            obs = self.y_test.values[t]
            history.append(obs)
        mape = mean_absolute_percentage_error(self.y_test, predictions) * 100
        rmse = (mean_squared_error(self.y_test, predictions)) ** (1 / 2)
        return mape, rmse, pd.Series(predictions, index=self.y_test.index)

    def plot_predictions(self):
        plot_predictions(self.y_test, self.predictions, self.cycle)


class ArimaPredictor(DataProcessing):
    def __init__(self, p, d, q, cycle=False):
        DataProcessing.__init__(self)
        self.x_train, self.x_test, self.y_train, self.y_test = self.ts_train_test_split()
        self.p = p
        self.q = q
        self.d = d
        self.cycle = cycle
        self.mape, self.rmse, self.predictions, self.garch_fitted, self.model_fit, self.predicted_et = self.model()

    def ts_train_test_split(self, n_test=12):
        df = self.df
        return df.iloc[:-n_test, 1:], df.iloc[-n_test:, 1:], df.iloc[:-n_test, 0], df.iloc[-n_test:, 0]

    def model(self):
        if self.cycle:
            self.x_train, self.x_test, self.y_train, self.y_test = cycle_ts_train_test_split(self.df, n_test=69)
            # self.p, self.q, self.d = 1, 0, 0
        history = [x for x in self.y_train.values]
        predictions = list()
        predicted_et = []
        for t in range(len(self.y_test)):
            model = sm.tsa.statespace.SARIMAX(history, order=(self.p, self.d, self.q), seasonal_order=(0, 0, 0, 0))
            model_fit = model.fit()
            output = model_fit.predict(start=len(self.y_train) + t, end=len(self.y_train) + 1 + t)
            yhat = output[0]
            predictions.append(yhat)
            obs = self.y_test[t]
            history.append(obs)
            garch = arch.arch_model(model_fit.resid, p=1, q=0)
            garch_fitted = garch.fit()
            garch_forecast = garch_fitted.forecast(horizon=1)
            predicted_et.append(garch_forecast.variance['h.1'].iloc[-1])
        mape = mean_absolute_percentage_error(self.y_test, predictions) * 100
        rmse = (mean_squared_error(self.y_test, predictions)) ** (1 / 2)
        return mape, rmse, pd.Series(predictions, index=self.y_test.index), garch_fitted, model_fit, predicted_et

    def ma_set(self):
        fig, axes = plt.subplots(2, 1, sharex=True)
        plot_acf(self.y_train, ax=axes[0], title="Autocorrelación Serie Original")
        plot_acf(self.y_train.diff().dropna(), ax=axes[1], title='Autocorrelación 1 Diferencia')
        plt.show()

    def ar_set(self):
        fig, axes = plt.subplots(2, 1, sharex=True)
        plot_pacf(self.y_train, ax=axes[0], title='Autocorrelación Parcial Serie original')
        plot_pacf(self.y_train.diff().dropna(), ax=axes[1], title='Autocorrelación Parcial 1 diferencia')
        plt.show()

    def plot_predictions(self):
        plot_predictions(self.y_test, self.predictions, self.cycle)

    def residual_diag(self):
        self.model_fit.plot_diagnostics(figsize=(7, 5))
        print(self.model_fit.summary())
        # line plot of residuals
        residuals = DataFrame(self.garch_fitted.std_resid)
        residuals.plot()
        plt.show()
        # density plot of residuals
        residuals.plot(kind='kde')
        plt.show()
        # summary stats of residuals
        print(sp.stats.shapiro(self.garch_fitted.std_resid))
        print("ARCH-LM test :", sms.diagnostic.het_arch(self.garch_fitted.std_resid, ddof=3))


class ARIMAXPredictor(DataProcessing):
    def __init__(self, p, d, q, cycle=False):
        DataProcessing.__init__(self)
        self.arimax_exvar = ['credito_ecuador(t-6)',
                             # 'inflacion(t-6)',
                             'tasa_pasiva(t-1)',
                             # 'credito_banco_pichincha_ecuador(t-5)',
                             # 'prestamo_ecuador(t-5)'
                             ]
        self.cycle = cycle
        self.x_train, self.x_test, self.y_train, self.y_test = self.ts_train_test_split()
        self.p = p
        self.q = q
        self.d = d
        # self.mape, self.rmse, self.predictions, self.model_fit = self.model()

    def ts_train_test_split(self, n_test=6):
        if self.cycle:
            df = self.df[:180]
            n_test = 69
        else:
            df = self.df
        X = df.loc[:, self.arimax_exvar].diff().fillna(value=0)
        return X.iloc[:-n_test, :], X.iloc[-n_test:, :], df.iloc[:-n_test, 0], df.iloc[-n_test:, 0]

    def model(self):
        history = self.y_train
        ex = np.array(self.x_train)
        predictions = []
        for t in range(len(self.x_test)):
            model = sm.tsa.statespace.SARIMAX(history, exog=ex, order=(self.p, self.d, self.q),
                                              seasonal_order=(0, 0, 0, 0))
            exog = self.x_test.iloc[[t]]
            obs = self.y_test[[t]]
            model_fit = model.fit()
            output = model_fit.predict(start=len(self.y_train) + t, end=len(self.y_train) + t, exog=exog)
            predictions.append(output[0])
            history.append(obs)
            ex = np.vstack((ex, exog))
        mape = mean_absolute_percentage_error(self.y_test, predictions) * 100
        rmse = (mean_squared_error(self.y_test, predictions)) ** (1 / 2)
        return mape, rmse, pd.Series(predictions, index=self.y_test.index), model_fit

    def plot_predictions(self):
        plot_predictions(self.y_test, self.predictions, self.cycle)

    def residual_diag(self):
        self.model_fit.plot_diagnostics(figsize=(7, 5))
        print(self.model_fit.summary())
        # line plot of residuals
        residuals = DataFrame(self.garch_fitted.std_resid)
        residuals.plot()
        plt.show()
        # density plot of residuals
        residuals.plot(kind='kde')
        plt.show()
        # summary stats of residuals
        print(sp.stats.shapiro(self.garch_fitted.std_resid))
        print("ARCH-LM test :", sms.diagnostic.het_arch(self.garch_fitted.std_resid, ddof=3))


class XgboostPredictor(DataProcessing):
    random.seed(28)
    np.random.seed(28)

    def __init__(self, cycle=False):
        DataProcessing.__init__(self)
        self.x_train, self.x_test, self.y_train, self.y_test = self.ts_train_test_split()
        self.cycle = cycle
        # self.mape, self.predictions, self.model_fit , self.best_iteration = self.model()
        self.mape, self.rmse, self.predictions, self.model_fit = self.model()

    def ts_train_test_split(self, n_test=6):
        df = self.df
        return df.iloc[:-n_test, 1:], df.iloc[-n_test:, 1:], df.iloc[:-n_test, 0], df.iloc[-n_test:, 0]

    def model(self):
        if self.cycle:
            self.x_train, self.x_test, self.y_train, self.y_test = cycle_ts_train_test_split(self.df, n_test=69)
        history = self.y_train
        ex = self.x_train
        predictions = []
        best_iteration = []
        for t in range(len(self.x_test)):
            # reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bylevel=1, colsample_bytree=1,
            #                        colsample_bynode=1, gamma=20, learning_rate=0.3, max_depth=4,
            #                        min_child_weight=10, n_estimators=160, reg_alpha=12,
            #                        reg_lambda=5, subsample=1,  random_state=28)
            reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bylevel=1, colsample_bytree=1,
                                   colsample_bynode=1, gamma=20, learning_rate=0.2, max_depth=3,
                                   min_child_weight=10, n_estimators=50, reg_alpha=12,
                                   reg_lambda=5, subsample=1, random_state=28)
            exog = self.x_test.iloc[[t]]
            obs = self.y_test[[t]]
            model_fit = reg.fit(ex,
                                history,
                                verbose=2,
                                # early_stopping_rounds=20,
                                # eval_set=[(ex, history), (exog, obs)],
                                # eval_metric=['rmse', 'mape'])
                                )
            output = model_fit.predict(exog)
            # best_iteration.append(model_fit.best_iteration)
            yhat = output[0]
            predictions.append(yhat)
            history = history.append(obs)
            ex = pd.concat([ex, exog], axis=0)
        mape = mean_absolute_percentage_error(self.y_test, predictions) * 100
        rmse = (mean_squared_error(self.y_test, predictions)) ** (1 / 2)
        # return mape, pd.Series(predictions, index=self.y_test.index), model_fit, best_iteration
        return mape, rmse, pd.Series(predictions, index=self.y_test.index), model_fit

    def plot_predictions(self):
        plot_predictions(self.y_test, self.predictions, self.cycle)

    def plot_importance(self):
        ax = plot_importance(self.model_fit, height=0.5)
        fig = ax.figure
        fig.set_size_inches(15, 15)
        plt.show()


class RandomForestPredictor(DataProcessing):
    def __init__(self, cycle=False):
        DataProcessing.__init__(self)
        self.x_train, self.x_test, self.y_train, self.y_test = self.ts_train_test_split()
        self.cycle = cycle
        self.mape, self.rmse, self.predictions, self.model_fit = self.model()

    def ts_train_test_split(self, n_test=6):
        df = self.df
        return df.iloc[:-n_test, 1:], df.iloc[-n_test:, 1:], df.iloc[:-n_test, 0], df.iloc[-n_test:, 0]

    def model(self):
        if self.cycle:
            self.x_train, self.x_test, self.y_train, self.y_test = cycle_ts_train_test_split(self.df, n_test=69)
        history = self.y_train
        ex = self.x_train.fillna(value=0)
        predictions = []
        for t in range(len(self.x_test)):
            reg = RandomForestRegressor(n_estimators=20, criterion="squared_error", max_features=0.8, max_leaf_nodes=15,
                                        max_samples=0.5, min_samples_leaf=3, min_samples_split=6, n_jobs=-1,
                                        random_state=28)
            exog = self.x_test.iloc[[t]]
            obs = self.y_test[[t]]
            model_fit = reg.fit(ex, history)
            output = model_fit.predict(exog)
            yhat = output[0]
            predictions.append(yhat)
            history = history.append(obs)
            ex = pd.concat([ex, exog], axis=0)
        mape = mean_absolute_percentage_error(self.y_test, predictions) * 100
        rmse = (mean_squared_error(self.y_test, predictions)) ** (1 / 2)
        return mape, rmse, pd.Series(predictions, index=self.y_test.index), model_fit

    def plot_predictions(self):
        plot_predictions(self.y_test, self.predictions, self.cycle)

    def plot_importance(self):
        plt.figure(figsize=(15, 15))
        feat_importances = pd.Series(self.model_fit.feature_importances_, index=self.x_train.columns)
        feat_importances.nlargest(100).sort_values(ascending=True).plot(kind='barh', title='Feature Importance')
        plt.xlabel("Gini-Impurity")
        plt.grid()
        plt.show()


class LSTM(DataProcessing):
    def __init__(self):
        DataProcessing.__init__(self)
        self.x_train, self.x_test, self.y_train, self.y_test = self.ts_train_test_split()
        # self.mape = self.model()

    def ts_train_test_split(self, n_test=6):
        df = self.df
        return df.iloc[:-n_test, 1:], df.iloc[-n_test:, 1:], df.iloc[:-n_test, 0], df.iloc[-n_test:, 0]
