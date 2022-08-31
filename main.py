from models import *

random.seed(28)
np.random.seed(28)

#-------------------------------------------------------------------------------
# BASELINE Model
#-------------------------------------------------------------------------------
model_b = BaselinePredictor()
model_b.plot_predictions()
model_b.mape

#-------------------------------------------------------------------------------
# ARIMA Model
#-------------------------------------------------------------------------------
arima = ArimaPredictor(2, 1, 1)
arima.ma_set()
arima.ar_set()
arima.plot_predictions()
arima.residual_diag()
arima.mape

#### AUTOARIMA CHECK
# ex = ["prestamo_ecuador(t-5)",
#       'prestamo_ecuador(t-6)'
#       'credito_ecuador(t-6)'
#       "simulador_de_credito_ecuador(t-4)",
#       "credito_quirografario_ecuador(t-2)",
#       "prestamo_quirografario_ecuador(t-2)",
#       "credito_banco_guayaquil_ecuador(t-6)",
#       "credito_ecuador(t-5)",
#       "prestamo_ecuador(t-6)",
#       "inflacion(t-5)",
#       "roe_sf(t-2)",
#       "tasa_pasiva(t-1)",
#       "tasa_pasiva(t-1)"]
# for x in range(len(ex)):
#     result = adfuller(arima.x_train.loc[:, ex[x]])
#     plt.plot(arima.x_train.loc[:, ex[x]])
#     plt.title(ex[x])
#     plt.show()
#     if result[1] > 0.01:
#         result = adfuller(arima.x_train.loc[:, ex[x]].diff().dropna())
#         plt.plot(arima.x_train.loc[:, ex[x]].diff().dropna())
#         plt.title(ex[x])
#         plt.show()
#         print('diff')
#     # result = adfuller(arima.x_train.iloc[:,0])
#     # print('ADF Statistic: %f' % result[0])
#     print(ex[x] + ' p-value: %f' % result[1])
#     # print('Critical Values:')
#
#
# model = pm.auto_arima(arima.y_train, start_p=0, start_q=0,
#                       test='adf',  # use adftest to find optimal 'd'
#                       max_p=2, max_q=2,  # maximum p and q
#                       m=12,  # frequency of series
#                       d=None,  # let model determine 'd'
#                       seasonal=True,  # No Seasonality
#                       start_P=0,
#                       D=0,
#                       trace=True,
#                       error_action='ignore',
#                       suppress_warnings=False,
#                       stepwise=False)
pass
#-------------------------------------------------------------------------------
# ARIMAX Model
#-------------------------------------------------------------------------------

arimax = ARIMAXPredictor(2, 1, 1)
arimax.mape
arima.plot_predictions()
#########LASSO REGRESSION FEATURE SELECTION
# from sklearn.linear_model import LassoCV, Lasso
# #
# laso_cv = TimeSeriesSplit(5, max_train_size=150, test_size=1)
# pipeline = Pipeline([
#                      ('scaler',StandardScaler()),
#                      ('model',Lasso())
# ])

# search = GridSearchCV(pipeline,
#                       {'model__alpha':np.arange(0.1,10,0.1)},
#                       cv = laso_cv, scoring="neg_mean_squared_error",verbose=3
#                       )
# #
# search.fit(arima.x_train.fillna(value=30), arima.y_train)
# search.best_params_
# coefficients = search.best_estimator_.named_steps['model'].coef_
# importance = np.abs(coefficients)
# np.array(importance)[importance > 0]
pass
#-------------------------------------------------------------------------------
# XGBOOST Model
#-------------------------------------------------------------------------------

xgboost = XgboostPredictor()
xgboost.mape
xgboost.plot_predictions()
xgboost.plot_importance()
# plt.savefig("p1")
#-------------------------------------------------------------------------------
# RANDOM FOREST Model
#-------------------------------------------------------------------------------

random_forest = RandomForestPredictor()
# xgboost.mape
# xgboost.plot_predictions()
# xgboost.plot_importance()

#-------------------------------------------------------------------------------
# RECURRENT NEURAL NETWORK (LSTM) Model
#-------------------------------------------------------------------------------

lstm = LSTM()
# lstm.mape
# lstm.plot_predictions()
# lstm.plot_importance()

