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
    'n_estimators': [140,130,300],
    'max_depth': [4,6],
    'min_child_weight': [10],
    'learning_rate': [0.3],
    'gamma': [20],
    'reg_lambda': [5],
    'subsample': [1],
    'colsample_bylevel': [0.7, 1],
    'colsample_bytree': [0.7, 1],
    'colsample_bynode': [0.7, 1],
    'reg_alpha': [10,12],
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

fit_params={"early_stopping_rounds":10,
            "eval_metric" : "mape",
            "eval_set" : [[x_test, y_test]]}

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

# optimal_params_df.to_excel("gridsearch2.xlsx")

#
random.seed(28)
np.random.seed(28)

#Predictions

reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bylevel=1, colsample_bytree=1,
                                   colsample_bynode=1, gamma=20, learning_rate=0.3, max_depth=4,
                                   min_child_weight=10, n_estimators=120, reg_alpha=12,
                                   reg_lambda=5, subsample=1,  random_state=28, seed=28)

reg.fit(x_train,
        y_train,
        verbose=2)

y_hat = reg.predict(x_test)
MAPE(y_test, y_hat)


predictions = []
besiter = []
for t in range(-6, 0, 1):
    x_train, x_test, y_train, y_test = x.iloc[:t, :], x.iloc[t:, :], y[:t], y[t:]
    reg.fit(x_train,
            y_train,
            verbose=2,
            early_stopping_rounds=20,
            eval_set=[(x_train, y_train), (x_test.iloc[:1], y_test.iloc[:1])],
            eval_metric=['rmse','mape']
            )
    y_hat = reg.predict(x_test.iloc[:1])

    # predictions.append(y_hat)
    predictions.append(y_hat)
    besiter.append(reg.best_iteration)
t = -12
    # x_train, x_test, y_train, y_test = x_encoded.iloc[:t, :], x_encoded.iloc[t:, :], y[:t], y[t:]
x_train, x_test, y_train, y_test = x.iloc[:t, :], x.iloc[t:, :], y[:t], y[t:]
# x_train, x_test, y_train, y_test = xgboost.x_train.iloc[:t, :], xgboost.x_test.iloc[t:, :], xgboost.y_train[:t], xgboost.y_test[t:]

# x_train, x_test, y_train, y_test = xgboost.x_train.iloc[:t, :], xgboost.x_test.iloc[t:, :], xgboost.y_train[:t], xgboost.y_test[t:]
#

mape_final = MAPE(y[-6:], predictions[6:])
mape_final = MAPE(xgboost.y_test, predictions)

MAPE(xgboost.y_test, xgboost.predictions)
mape_final

results = reg.evals_result()
# Feature Importance
ax = plot_importance(reg, height=0.5)
fig = ax.figure
fig.set_size_inches(60, 60)
plt.show()
print(reg.feature_importances_)

optimal_params_df.to_excel("gridsearch3.xlsx")

# xgboost.df.to_excel("base_hogar_hypertunning.xlsx")