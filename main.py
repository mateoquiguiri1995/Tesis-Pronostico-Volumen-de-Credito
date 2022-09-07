from models import *
# from models import BaselinePredictor

random.seed(28)
np.random.seed(28)

#-------------------------------------------------------------------------------
# BASELINE Model
#-------------------------------------------------------------------------------
model_b: BaselinePredictor = BaselinePredictor(True)
model_b.plot_predictions()
model_b.mape

#-------------------------------------------------------------------------------
# ARIMA Model
#-------------------------------------------------------------------------------
arima = ArimaPredictor(2, 1, 1, cycle=True)
arima.ma_set()
arima.ar_set()
arima.plot_predictions()
arima.residual_diag()
arima.mape

pass
#-------------------------------------------------------------------------------
# ARIMAX Model
#-------------------------------------------------------------------------------

arimax = ARIMAXPredictor(2, 1, 1, cycle=False)
# arimax.mape
# arima.plot_predictions()

#-------------------------------------------------------------------------------
# XGBOOST Model
#-------------------------------------------------------------------------------

xgboost = XgboostPredictor(cycle=False)
xgboost.mape
xgboost.plot_predictions()
xgboost.plot_importance()
# plt.savefig("p1")
#-------------------------------------------------------------------------------
# RANDOM FOREST Model
#-------------------------------------------------------------------------------

random_forest = RandomForestPredictor(cycle=False)
random_forest.mape
random_forest.plot_predictions()
random_forest.plot_importance()

#-------------------------------------------------------------------------------
# RECURRENT NEURAL NETWORK (LSTM) Model
#-------------------------------------------------------------------------------

lstm = LSTM()
# lstm.mape
# lstm.plot_predictions()
# lstm.plot_importance()

