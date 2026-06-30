# -*- coding: utf-8 -*-
"""

@author: jpshlima
"""

# load best anatel lstm model, load base for predictions and prepare data for classification

from keras.models import model_from_json
def loadTrainedNN():
    # loading saved parameters 
    saved_net = open('anatel_gru_rsrp.json', 'r')
    struct_net = saved_net.read()
    saved_net.close()
    regressor = model_from_json(struct_net)
    regressor.load_weights('anatel_gru_rsrp.weights.h5')
    return (regressor)


def getPredictions(df):
    prevs = []

    # get only RSRP values from 1 UE as time series
    rsrp = df['RSRP'].values
    rsrp = rsrp.reshape(-1, 1)

    from sklearn.preprocessing import MinMaxScaler

    # apply MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    rsrp_norm = scaler.fit_transform(rsrp)

    # train and test split
    rsrptest = rsrp_norm[8006:8896, :]
    ho_trig = df['ho_trig'].values

    # preparing inputs for test
    inputs = rsrp_norm[len(rsrp_norm) - len(rsrptest) - 100:]
    inputs = inputs.reshape(-1, 1)

    x_test = []
    for i in range(100, inputs.size):
        x_test.append(inputs[i-100:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction = regressor.predict(x_test)

    # convert model output to simple 1D numeric vector
    prediction = np.asarray(prediction).reshape(-1, 1)

    # undo normalization
    prediction = scaler.inverse_transform(prediction)

    # make prediction like: [12.5, 13.2, 14.1, ...]
    prediction = prediction.ravel()

    aux = np.zeros((len(prediction), 2))

    # first column: predicted RSRP
    aux[:, 0] = prediction

    # second column: handover label
    label_start = len(rsrp_norm) - len(rsrptest)
    aux[:, 1] = ho_trig[label_start:label_start + len(prediction)]

    prevs.append(aux)
    prevs = np.vstack(prevs)

    return prevs

import numpy as np
import pandas as pd
from pathlib import Path

# First, read and prepare RSRP data
files = ['drive_test_measurements01.csv', 'drive_test_measurements02.csv', 'drive_test_measurements03.csv']
base_dir=Path(__file__).resolve().parent
data_path=base_dir.parent / "real_data"
df = pd.concat((pd.read_csv(data_path / f) for f in files))
df.drop(df.columns[[0,1,2,4,5,7,8,9,10]], axis=1, inplace=True)
df['ho_trig'] = 0
df.reset_index(drop=True, inplace=True)

for i in range(2, df.shape[0] - 1):
    if df.loc[i, 'PCI'] != df.loc[i - 1, 'PCI']:
        df.loc[i, 'ho_trig'] = 1



regressor = loadTrainedNN()

prevs = getPredictions(df)

concatbases = []
# loop for filling variable
x_test = []
for i in range (50, prevs.shape[0]):
    x_test.append(prevs[i-50:i, 0])
x_test = np.array(x_test)        

classification_base = pd.DataFrame(x_test)
classification_base['label'] = prevs[49:prevs.shape[0]-1, 1]
concatbases.append(classification_base)
concatbases = np.vstack(concatbases)
concatbases2 = pd.DataFrame(concatbases)
concatbases2.to_csv('anatel_concatbases_gru.csv', index=False)


