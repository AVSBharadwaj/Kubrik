import requests
import pandas as pd
import scipy
import numpy 
import sys
from sklearn.linear_model import ElasticNet,LinearRegression

TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    train_data=pd.read_csv(TRAIN_DATA_URL,header=None)
    
    train_area=train_data.loc[0]
    train_price=train_data.loc[1]
    train_area=train_area.iloc[1:]
    
    train_price=train_price.iloc[1:]
    train_area=train_area.to_numpy()
    train_price=train_price.to_numpy()
   
    train_area=train_area.reshape((train_area.shape[0],1))
    train_price=train_price.reshape((train_price.shape[0],1))
    
    El=ElasticNet(alpha=0.5,l1_ratio=1,normalize=False)
    El.fit(train_area,train_price)
    area=area.reshape((area.shape[0],1))
    
    return El.predict(area)
    
    ...


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
