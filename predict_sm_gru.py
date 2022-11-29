import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import tensorflow as tf
from tensorflow import keras
import os
from time import process_time
import time
from sklearn.metrics import r2_score

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# set seed
tf.random.set_seed(786)

# global dataclass with hyperparameters and plot data
@dataclass
class G:
    FILENAME = 'data/merged.xlsx'
    WINDOW_SIZE = 50
    PRED_SIZE = 10
    SPLIT_FRAC = 0.85
    SHUFFLE_BUFFER = 20
    BATCH_SIZE = 50
    GRU_WIDTH = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 200
    DROPOUT = 0.3
    LAYERS = 3_1

# normalization function
def normalize(data):
    # calculate dataset mean and standard deviation
    mean = data.mean()
    std = data.std()

    comb = pd.concat([mean, std], axis=1)
    comb.columns=['mean', 'std']
    comb.to_excel('data/norm.xlsx', sheet_name='mean_std')

    # get column names
    col = data.columns

    # normalise dataset with previously calculated values
    data_norm =(data[col] - mean)/std

    return data_norm

# read data function
def parse_data_from_file(filename):
    # load data
    df = pd.read_excel(filename, sheet_name='delta')
    df_sm = pd.read_excel(filename, sheet_name='Sheet1')
    # # remove period with incorrect data
    # df = df[~(df['datum'] > '2020-10-03')]

    # set datum to index
    df.set_index('weather_date', inplace=True)
    df_sm.set_index('weather_date', inplace=True)
    # # sort data from past to present
    # df = df.iloc[::-1]
    df_sm = df_sm['soil_15']
    # normalize data
    df_norm = normalize(df)
    df_merged = pd.merge(df_sm, df_norm, how='inner', left_on='weather_date', right_on='weather_date')
    # convert to numpy array
    df_np = df_merged.to_numpy()
    # df_norm_np = df_norm.to_numpy()
    # df_sm_np = df_sm.to_numpy()[:,0]

    return df_np

# create windowed dataset
def windowed_data(data, window_size, pred_size):
    # get data dimensions
    dim = data.shape

    # create empty arrays for X and y
    X = np.zeros([dim[0]-window_size-pred_size, window_size+pred_size, dim[1]-2])
    y = np.zeros([dim[0]-window_size-pred_size, pred_size+1, 2])

    # create windowed data
    for i in range(dim[0]-window_size-pred_size):
        X[i,:,:] = data[i:(i+window_size+pred_size),1:]
        y[i,:] = data[(i+window_size):(i+window_size+pred_size),0]

    return X, y

def split_data(X,y,split_frac):
    split = int(X.shape[0]*split_frac)
    X_train = X[:split,:,:]
    y_train = y[:split,:]
    X_test = X[split:,:,:]
    y_test = y[split:,:]

    return X_train, y_train, X_test, y_test

# batch and shuffle the data
def data_batch(X, y, shuffle_buffer, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(1)

    return ds

# model architecture
def create_model():

    inputs = keras.Input(shape=((G.WINDOW_SIZE+G.PRED_SIZE), 3))
    x = keras.layers.GRU(G.GRU_WIDTH, dropout=G.DROPOUT, return_sequences=True)(inputs)
    x = keras.layers.GRU(G.GRU_WIDTH, dropout=G.DROPOUT, return_sequences=True)(x)
    # x = keras.layers.GRU(G.GRU_WIDTH, dropout=G.DROPOUT, return_sequences=True)(x)
    # x = keras.layers.GRU(G.GRU_WIDTH, dropout=G.DROPOUT, return_sequences=True)(x)
    x = keras.layers.GRU(G.GRU_WIDTH, dropout=G.DROPOUT)(x)
    # x = keras.layers.Dense(64, activation="tanh")(x)
    x = keras.layers.Dense(32, activation="tanh")(x)
    # x = keras.layers.Dense(64, activation="sigmoid")(x)
    outputs = keras.layers.Dense(G.PRED_SIZE)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

@tf.function
def train_step(X,y_true):
    with tf.GradientTape() as tape:

        y_pred = model(X)

        # loss_value = tf.compat.v1.losses.huber_loss(y_true, y_pred)
        # loss_value = tf.keras.metrics.mean_squared_error(y_true, y_pred)
        loss_value = tf.keras.metrics.mean_absolute_error(y_true, y_pred)

    # calculate gradient with respect to weights of the network
    grads = tape.gradient(loss_value, model.trainable_weights)

    # update model
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # loss_value = tf.reduce_sum(loss_value)/len(loss_value)

    return loss_value

@tf.function
def test_step(X,y_true):

    y_pred = model(X)

    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=tf.float32)

    # loss_value = tf.compat.v1.losses.huber_loss(y_true, y_pred)
    # loss_value = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    loss_value = tf.keras.metrics.mean_absolute_error(y_true, y_pred)

    loss_value = tf.reduce_sum(loss_value)/len(loss_value)

    return y_pred, loss_value

# compute mse and mae
def compute_metrics(true_series, forecast):

    mse_all = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
    mae_all = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()
    r2_all = r2_score(true_series, forecast)

    mse_all = sum(mse_all) / len(mse_all)
    mae_all = sum(mae_all) / len(mae_all)

    mse = np.zeros(G.PRED_SIZE)
    mae = np.zeros(G.PRED_SIZE)
    r2 = np.zeros(G.PRED_SIZE)

    for i in range(G.PRED_SIZE):
        mse[i] = tf.keras.metrics.mean_squared_error(true_series[:,i], forecast[:,i]).numpy()
        mae[i] = tf.keras.metrics.mean_absolute_error(true_series[:,i], forecast[:,i]).numpy()
        r2[i] = r2_score(true_series[:,i], forecast[:,i])

    return mse, mae, r2, mse_all, mae_all, r2_all

# laod data
data = parse_data_from_file(G.FILENAME)

# create windowed dataset
X, y = windowed_data(data, G.WINDOW_SIZE, G.PRED_SIZE)

# split dataset in train and test set
X_train, y_train, X_test, y_test = split_data(X,y,G.SPLIT_FRAC)

# batch train data
train_data = data_batch(X_train, y_train, G.SHUFFLE_BUFFER, G.BATCH_SIZE)

# initialize model
model = create_model()

# set optimizer and learning rate
optimizer=tf.keras.optimizers.Adam(learning_rate=G.LEARNING_RATE)

save_title = '_gru_'+str(G.EPOCHS)+'_'+str(G.LAYERS)+'_'+str(G.GRU_WIDTH)+'_dr_'+str(G.DROPOUT)+'_lr_'+str(G.LEARNING_RATE)+'_pred_'+str(G.PRED_SIZE)+'_win_'+str(G.WINDOW_SIZE)

# training loops
train_loss_plot = []
test_loss_plot = []

for epoch in range(1,G.EPOCHS+1):

    print("\n--------------------- \nStart of epoch %d" % (epoch,))
    tic_p = time.process_time()
    tic_c = time.perf_counter()
    loss = 0

    for step, (X_batch_train, y_batch_train) in enumerate(train_data):

        X_batch_train = tf.cast(X_batch_train, dtype=tf.float32)
        y_batch_train = tf.cast(y_batch_train, dtype=tf.float32)

        loss += train_step(X_batch_train, y_batch_train)

    train_loss = loss/len(train_data)

    toc_p = time.process_time()
    toc_c = time.perf_counter()

    timer_p = toc_p - tic_p
    timer_c = toc_c - tic_c

    print('\nProcess time per epoch = ', timer_p)
    print('\nClock time per epoch =   ', timer_c)

    y_pred, test_loss = test_step(X_test, y_test)

    print("\nTrain Loss = %.4f" % (float(train_loss)))
    print("\nTest_Loss = %.4f" % (float(test_loss)))

    train_loss_plot.append(train_loss)
    test_loss_plot.append(test_loss)

    np.savetxt('gru/loss_history/train_loss_'+save_title+'.csv', train_loss_plot)
    np.savetxt('gru/loss_history/test_loss_'+save_title+'.csv', test_loss_plot)

# get predictions
y_pred_test = model(X_test)
y_pred_all = model(X)

# get mse and mea
mse, mae, r2, mse_all, mae_all, r2_all = compute_metrics(y_test, y_pred_test)

# make plot title
plot_title = 'Epochs = '+str(G.EPOCHS)+', pred_days = '+str(G.PRED_SIZE)+'\n Layers = '+str(G.LAYERS)+', Width = '+str(G.GRU_WIDTH)+', Learning_rate = '+str(G.LEARNING_RATE)+', Dropout = '+str(G.DROPOUT)+'\n MSE = '+str(mse_all)+', MAE = '+str(mae_all)

if not os.path.exists('gru/predictions/'+save_title):
   os.makedirs('gru/predictions/'+save_title)

if not os.path.exists('gru/model/'+save_title):
   os.makedirs('gru/model/'+save_title)

model.compile()
model.save('gru/model/'+save_title)

X_plot_test = np.arange(0,len(y_test))
error_plot_test = np.arange(1,G.PRED_SIZE+1)

# plot loss loss_history
plt.figure(figsize=(12,8))
plt.plot(np.arange(epoch), train_loss_plot, label='train_loss')
plt.plot(np.arange(epoch), test_loss_plot, label='test_loss')
plt.yscale('log')
plt.xlabel('epoch', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.legend(fontsize=14)
plt.title(plot_title)
plt.savefig('gru/loss_history/loss'+save_title+'.png')
plt.close()

for i in range(G.PRED_SIZE):
    plt.figure(figsize=(20,10))
    plt.plot(X_plot_test, y_test[:,i], label='Soil moisture - true', color='black', lw = 0.5)
    plt.plot(X_plot_test, y_pred_test[:,i], label='Soil moisture - predicted', color='blue', lw = 0.5)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Soil moisture (normalized)')
    plt.title('Soil moisture - day '+str(i+1)+'\n'+plot_title)
    plt.savefig('gru/predictions/'+save_title+'/test_day_'+str(i+1)+'.png')
    plt.close()

fig, ax1 = plt.subplots(figsize=(10,5))
ax2 = ax1.twinx()
lns1 = ax1.plot(error_plot_test, mse, label='MSE', color='black', lw = 0.5)
lns2 = ax1.plot(error_plot_test, mae, label='MAE', color='blue', lw = 0.5)
lns3 = ax2.plot(error_plot_test, r2, label='R2', color='red', lw=0.5)
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs)
ax1.set_xlabel('Day')
ax1.set_ylabel('Error')
ax2.set_ylabel('R2', color='red')
plt.title('Error '+plot_title)
plt.savefig('gru/predictions/'+save_title+'/test_error.png')
plt.close()

breakpoint()
