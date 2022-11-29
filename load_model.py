import pandas as pd
import tensorflow as tf
import numpy as np

window = 50 # back in time
pred = 10 # days predicted

# import normalisation data and model
norm = pd.read_excel('nn_model/norm.xlsx', sheet_name='mean_std')
model = tf.keras.models.load_model('nn_model/model')
# seperate mean and std
mean = norm['mean'].to_numpy()
std = norm['std'].to_numpy()
# read data for prediction
df = pd.read_excel('data/merged.xlsx', sheet_name='Sheet1')
# set datum to index
df.set_index('weather_date', inplace=True)
# get deltas
df_D = df.diff()
# drop empty row
df_D.dropna(inplace=True)
# normalize deltas
col = df_D.columns
df_D_norm = (df_D[col] - mean)/std
# find start and end dates of input for latest prediction
start = df_D_norm.index[-(window+pred)]
y0_inx = df_D_norm.index[-pred]
end = df_D_norm.index[-1]
# get input for prediction
input_data = df_D_norm[start:end].to_numpy()
# create empty array for input
X = np.zeros([1, window+pred, input_data.shape[1]-1])
# get last known soil moisture
y0 = df.loc[y0_inx][0]
# write data to input array
X[0,:,:] = input_data[:,1:]
# get prediction of delta
dy_norm = model(X)
# unnormalize delta's
dy_d = dy_norm*std[0] + mean[0]
# get cumulative values of predictions
dy = tf.cumsum(dy_d, axis=1)

# add delta's to soil moisture at day 0 to get predicted soil moisture
y = dy+y0

breakpoint()
