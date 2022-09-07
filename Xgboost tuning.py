import random
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import plot_importance
from xgboost import plot_tree
from processing import *
from models import *
import missingno as msno
from sklearn.ensemble import RandomForestRegressor

base_hogares = pd.read_excel("data/base_hogares.xlsx")
base_hogares['fecha'] = pd.to_datetime(base_hogares['fecha'], format='%Y-%m-%d').dt.date
base_hogares = base_hogares.set_index('fecha')
#
# # #######################################################################
# # # Crédito Hogares
# # #######################################################################
# #
# # # DATA PREPARING FOR XGBOOST: Format
# #
# # # First Part: The columns of data we will use to make regression
x = base_hogares.drop("credito_hogares", axis=1).copy()
# x.head()
# #
# # # Second Part: The column we want to predict
y = base_hogares['credito_hogares'].copy()
# y.head()
# #
# # # How the data looks when we use One Hot encoding. NOT SUITABLE FOR LINEAR AND LOGISTIC REGRESSION
# pd.get_dummies(x, columns=["year"]).head()
# #
# # # DATA PREPARING: One Hot Encoding or make dummies variables
# x = pd.get_dummies(x, columns=["year", 'month'])
#
xgboost = XgboostPredictor()

# BUILD A PRELIMINARY XGBOOST MODEL: Split the data
t = -6
# x_train, x_test, y_train, y_test = x.iloc[:t, :], x.iloc[t:, :], y[:t], y[t:]
x_train, x_test, y_train, y_test = xgboost.x_train, xgboost.x_test, xgboost.y_train, xgboost.y_test
x = pd.concat([x_train, x_test], axis=0)
y = y_train.append(y_test)
#
param_grid = {
    'n_estimators': [140, 130, 300],
    'max_depth': [4, 6],
    'min_child_weight': [10],
    'learning_rate': [0.3],
    'gamma': [20],
    'reg_lambda': [5],
    'subsample': [1],
    'colsample_bylevel': [0.7, 1],
    'colsample_bytree': [0.7, 1],
    'colsample_bynode': [0.7, 1],
    'reg_alpha': [10, 12],
    'random_state': [28],
    'n_jobs': [-1]

}

# param_grid = {
#     'n_estimators': [100,300],
#     'max_depth': [3, 4, 5],
#     'min_child_weight': [ 3,5,7],
#     'learning_rate': [0.01,0.1,0.3],
#     'gamma': [0.25, 0.5, 1],
#     'reg_lambda': [5, 10, 20],
#     'subsample': [0.9,1],
#     'colsample': [0.5,0.7,0.9,1],
#     'reg_alpha': [5, 10]
#     # 'reg_lambda': [1],
# }
# param_grid = {
#     'n_estimators': [20,25,50,75,100],
#     'max_depth': [3],
#     'min_child_weight': [7],
#     'learning_rate': [0.3,0.2],
#     'gamma': [0.15,0.25,1],
#     'reg_lambda': [1,5],
#     'subsample': [1, 0.9],
#     'colsample': [0.5,0.7,0.9,1],
#     'reg_alpha': [10,20]
#     # 'reg_lambda': [1],
# }

# param_grid = {
#     'n_estimators': [30,70,150],
#     'max_depth': [3, 4, 5],
#     'min_child_weight': [5, 7, 9 ],
#     'learning_rate': [0.05,0.15,0.3],
#     'gamma': [0.25, 0.5, 1],
#     'reg_lambda': [5, 10],
#     'subsample': [0.8,1],
#     'colsample_bytree': [0.5,0.7,1],
#     'colsample_bylevel': [0.5,1],
#     'reg_alpha': [1]
# }

n_splits = 6
tscv = TimeSeriesSplit(n_splits, test_size=1)
#
# for fold, (train_index, test_index) in enumerate(tscv.split(x_train)):
#     print("Fold: {}".format(fold))
#     print("TRAIN indices:", train_index, "\n", "TEST indices:", test_index)
#     print("\n")
random.seed(28)
np.random.seed(28)
# param_grid = {
#     'colsample_bylevel': [0.7,1],
#     'colsample_bytree': [0.7,1],
#     'colsample_bynode': [0.7,1],
#     'gamma': [1,10],
#     'learning_rate': [0.3],
#     'max_depth': [6],
#     'min_child_weight': [10],
#     'n_estimators': [140,160],
#     'reg_alpha': [10],
#     'reg_lambda': [5],
#     'subsample': [1],
#     'random_state': [28],
#     'n_jobs': [1]
#
# }

fit_params = {"early_stopping_rounds": 10,
              "eval_metric": "mape",
              "eval_set": [[x_test, y_test]]}

optimal_params = GridSearchCV(
    estimator=xgb.XGBRegressor(objective='reg:squarederror',
                               random_state=28),
    param_grid=param_grid,
    # fit_params=fit_params,
    scoring='neg_mean_absolute_percentage_error',
    verbose=2,
    n_jobs=1,
    # random_state=28,
    cv=tscv)

# optimal_params.fit(x_train, y_train, **fit_params)
optimal_params.fit(x_train, y_train)
optimal_params_df = pd.DataFrame(optimal_params.cv_results_)
# optimal_params.best_params_


optimal_params_df.to_excel("gridsearch3.xlsx")

# xgboost.df.to_excel("base_hogar_hypertunning.xlsx")

# -------------------------------------------------------------------------------
# RANDOM FOREST TUNING
# -------------------------------------------------------------------------------
random_forest = RandomForestPredictor()
x_train, x_test, y_train, y_test = random_forest.x_train, random_forest.x_test, random_forest.y_train, random_forest.y_test

n_splits = 6
tscv = TimeSeriesSplit(n_splits, test_size=1)

n_estimators = [20, 50, 100, 200, 300]  # number of trees in the random forest
max_features = ['0.2', '0.4', '0.8', '1', 'sqrt']  # number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(5, 50, num=10)]  # maximum number of levels allowed in each decision tree
min_samples_split = [2, 6, 10]  # minimum sample number to split a node
min_samples_leaf = [3, 5, 7]  # minimum sample number that can be stored in a leaf node
bootstrap = [True, False]  # method used to sample data points
max_leaf_nodes = [6, 10, 15]
max_samples = [1]
n_jobs = [-1]
random_state = [28]

grid = {'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap,
        'n_jobs': n_jobs,
        'max_leaf_nodes': max_leaf_nodes,
        'max_samples': max_samples,
        'random_state': random_state
        }

optimal_params = GridSearchCV(
    estimator=RandomForestRegressor(criterion='squared_error'),
    param_grid=grid,
    scoring='neg_mean_absolute_percentage_error',
    verbose=2,
    cv=tscv)

optimal_params.fit(x_train, y_train)
optimal_params_df = pd.DataFrame(optimal_params.cv_results_)
model_b.df.to_excel("base_hogar_random_forest.xlsx")

# -------------------------------------------------------------------------------
# MISSING VALUES CHECK
# -------------------------------------------------------------------------------

model_b = BaselinePredictor()
msno.bar(model_b.orig_df)
plt.show()
msno.matrix(model_b.orig_df)
plt.show()
msno.heatmap(model_b.orig_df)
plt.show()

# -------------------------------------------------------------------------------
# AUTOARIMA CHECK
# -------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------
# LASSO REGRESSION FEATURE SELECTION FOR ARIMAX
# -------------------------------------------------------------------------------

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