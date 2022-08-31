import random
import arch
import pandas as pd  # load and manipulate data and hot encoding
import numpy as np  # calculate the mean and standard deviation
import xgboost as xgb  # Xgboost stuff
from sklearn.model_selection import GridSearchCV  # Cross-validation
import statsmodels.api as sm
import statsmodels.stats as sms
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from xgboost import plot_importance
import scipy as sp
from processing import *
from sklearn.model_selection import TimeSeriesSplit


def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class BaselinePredictor(DataProcessing):
    def __init__(self):
        DataProcessing.__init__(self)
        self.x_train, self.x_test, self.y_train, self.y_test = self.ts_train_test_split()
        self.mape, self.predictions = self.model()

    def ts_train_test_split(self, n_test=12):
        df = self.df
        return df.iloc[:-n_test, 1:], df.iloc[-n_test:, 1:], df.iloc[:-n_test, 0], df.iloc[-n_test:, 0]

    def model(self):
        history = list(self.y_train.values)
        predictions = list()
        for t in range(len(self.y_test)):
            output = history[-1]
            predictions.append(output)
            obs = self.y_test.values[t]
            history.append(obs)
        mape = MAPE(self.y_test, predictions)
        return (mape), pd.Series(predictions, index=self.y_test.index)

    def plot_predictions(self):
        plt.plot(self.y_test, label='Observado')
        plt.plot(self.predictions, label='Pronosticado')
        plt.legend()
        plt.show()


class ArimaPredictor(DataProcessing):
    def __init__(self, p, d, q):
        DataProcessing.__init__(self)
        self.x_train, self.x_test, self.y_train, self.y_test = self.ts_train_test_split()
        self.p = p
        self.q = q
        self.d = d
        self.mape, self.predictions, self.garch_fitted, self.model_fit, self.predicted_et = self.model()

    def ts_train_test_split(self, n_test=12):
        df = self.df
        return df.iloc[:-n_test, 1:], df.iloc[-n_test:, 1:], df.iloc[:-n_test, 0], df.iloc[-n_test:, 0]

    def model(self):
        history = [x for x in self.y_train.values]
        predictions = list()
        predicted_et = []
        for t in range(len(self.y_test)):
            model = sm.tsa.statespace.SARIMAX(history, order=(self.p, self.d, self.q), seasonal_order=(0, 0, 0, 0))
            # model = sm.tsa.arima.ARIMA(history, order=(self.p, self.d, self.q), seasonal_order=(0, 0, 0, 0))
            model_fit = model.fit()
            output = model_fit.predict(start=len(self.y_train) + t, end=len(self.y_train) + 1 + t)
            yhat = output[0]
            predictions.append(yhat)
            obs = self.y_test[t]
            history.append(obs)
            # print(model_fit.summary())
            garch = arch.arch_model(model_fit.resid, p=1, q=0)
            garch_fitted = garch.fit()
            # print(garch_fitted.summary())
            garch_forecast = garch_fitted.forecast(horizon=1)
            predicted_et.append(garch_forecast.variance['h.1'].iloc[-1])
        mape = MAPE(self.y_test, predictions)

        return mape, pd.Series(predictions, index=self.y_test.index), garch_fitted, model_fit, predicted_et

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
        plt.plot(self.y_test, label='Observado')
        plt.plot(self.predictions, label='Pronosticado')
        plt.legend()
        plt.show()

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
    def __init__(self, p, d, q):
        DataProcessing.__init__(self)
        # self.arimax_exvar = ["credito_banco_pichincha_ecuador(t-5)",
        #                      "simulador_de_credito_ecuador(t-4)",
        #                      "credito_quirografario_ecuador(t-2)",
        #                      "prestamo_quirografario_ecuador(t-2)",
        #                      "credito_banco_guayaquil_ecuador(t-6)",
        #                      "credito_ecuador(t-5)",
        #                      "prestamo_ecuador(t-6)",
        #                      "inflacion(t-5)",
        #                      "roe_sf(t-2)",
        #                      "tasa_pasiva(t-1)",
        #                      "tasa_pasiva(t-1)"]
        self.arimax_exvar = ['credito_ecuador(t-6)',
                             'inflacion(t-6)',
                             'tasa_pasiva(t-1)',
                             'credito_banco_pichincha_ecuador(t-5)',
                             'prestamo_ecuador(t-5)']
        self.x_train, self.x_test, self.y_train, self.y_test = self.ts_train_test_split()
        self.p = p
        self.q = q
        self.d = d
        # self.mape = self.model()
        self.mape, self.history, self.predictions, self.ex = self.model()

    def ts_train_test_split(self, n_test=6):
        df = self.df
        X = df.loc[:, self.arimax_exvar].diff().dropna()
        return X.iloc[:-n_test, :], X.iloc[-n_test:, :], df.iloc[1:-n_test, 0], df.iloc[-n_test:, 0]

    def model(self):
        history = [x for x in self.y_train]
        # ex = np.array(self.x_train.loc[:, self.arimax_exvar])
        ex = np.array(self.x_train)
        predictions = list()
        for t in range(len(self.x_test)):
            model = sm.tsa.statespace.SARIMAX(history, exog=ex, order=(self.p, self.d, self.q),
                                              seasonal_order=(0, 0, 0, 0))
            model_fit = model.fit()
            exog = np.array(self.x_test)[t]
            output = model_fit.predict(start=len(self.x_train) + t, end=len(self.x_train) + t, exog=exog)
            predictions.append(output[0])
            obs = self.y_test[t]
            history.append(obs)
            ex = np.vstack((ex, exog))
        mape = MAPE(self.y_test, predictions)
        return (mape), history, predictions, ex


class XgboostPredictor(DataProcessing):
    random.seed(28)
    np.random.seed(28)

    def __init__(self):
        DataProcessing.__init__(self)
        self.x_train, self.x_test, self.y_train, self.y_test = self.ts_train_test_split()
        # self.mape, self.predictions, self.model_fit , self.best_iteration = self.model()
        self.mape, self.predictions, self.model_fit = self.model()

    def ts_train_test_split(self, n_test=6):
        df = self.df
        return df.iloc[:-n_test, 1:], df.iloc[-n_test:, 1:], df.iloc[:-n_test, 0], df.iloc[-n_test:, 0]

    def model(self):
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
        mape = MAPE(self.y_test, predictions)
        # return mape, pd.Series(predictions, index=self.y_test.index), model_fit, best_iteration
        return mape, pd.Series(predictions, index=self.y_test.index), model_fit

    def plot_predictions(self):
        plt.plot(self.y_test, label='Observado')
        plt.plot(self.predictions, label='Pronosticado')
        plt.legend()
        plt.show()

    def plot_importance(self):
        ax = plot_importance(self.model_fit, height=0.5)
        fig = ax.figure
        fig.set_size_inches(30, 30)
        plt.show()


class RandomForestPredictor(DataProcessing):
    def __init__(self):
        DataProcessing.__init__(self)
        self.x_train, self.x_test, self.y_train, self.y_test = self.ts_train_test_split()
        # self.mape = self.model()

    def ts_train_test_split(self, n_test=6):
        df = self.df
        return df.iloc[:-n_test, 1:], df.iloc[-n_test:, 1:], df.iloc[:-n_test, 0], df.iloc[-n_test:, 0]

    # def model(self):
    #     history = [x for x in self.y_train.values]
    #     ex = np.array(self.x_train)
    #     predictions = list()
    #     for t in range(len(self.test)):
    #         reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=120, learning_rate=0.3, seed=42,
    #                                min_child_weight=7, reg_lambda=5,
    #                                colsample=1, gamma=1, max_depth=3, subsample=1)
    #
    #         exog = np.array(self.x_test)[t]
    #         model_fit = reg.fit(ex,
    #                             history,
    #                             verbose=True,
    #                             early_stopping_rounds=30,
    #                             eval_set=[(ex, history), (self.x_test[t], self.y_test[t])],
    #                             eval_metric=['rmse', 'mape'])
    #         output = model_fit.predict([exog])
    #         yhat = output[0]
    #         predictions.appendu(yhat)
    #         obs = self.y_test[t]
    #         history.append(obs)
    #         ex = np.vstack((ex, exog))
    #     mape = MAPE(self.x_test, predictions)
    #     return (mape)


class LSTM(DataProcessing):
    def __init__(self):
        DataProcessing.__init__(self)
        self.x_train, self.x_test, self.y_train, self.y_test = self.ts_train_test_split()
        # self.mape = self.model()

    def ts_train_test_split(self, n_test=6):
        df = self.df
        return df.iloc[:-n_test, 1:], df.iloc[-n_test:, 1:], df.iloc[:-n_test, 0], df.iloc[-n_test:, 0]
