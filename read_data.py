import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate
from scipy.signal import correlation_lags
from sklearn.decomposition import PCA

# normalization function
def normalize(data):
    # calculate dataset mean and standard deviation
    mean = data.mean()
    std = data.std()

    # get column names
    col = data.columns

    # normalise dataset with previously calculated values
    data_norm =(data[col] - mean)/std

    return data_norm

filename = 'data/merged.xlsx'

df = pd.read_excel(filename, sheet_name='delta')

## remove wrong data
# df = df[~(df['datum'] > '2020-10-03')]

# set date as index
df.set_index('weather_date', inplace=True)

# # sort data from past to present
# df = df.iloc[::-1]

df = normalize(df)
# store column names
col = df.columns

# define label and features
y = df[col[0]]
x = df[col[1:]]

# pca_comp = 5
# # PCA analyis
# pca_x = PCA(n_components=pca_comp)
# p_x = pca_x.fit_transform(x)
# print('Explained variation per principal component: {}'.format(pca_x.explained_variance_ratio_))
#
# plt.figure(figsize=(15,20))
# for i in range(pca_comp):
#      plt.subplot(pca_comp, 1, i+1)
#      plt.scatter(y,p_x[:,i], alpha=0.5)
#      plt.title('PC_'+str(i+1))
#      plt.grid()
# plt.suptitle('PCA')
# plt.savefig('data/pca_d.png')
# plt.close()

# calculate lag per feature
lag = np.zeros(len(col)-1)

for i in range(1,len(col)):
    x = df[col[i]]
    correlation = correlate(x, y, mode='full')
    lags = correlation_lags(x.size, y.size, mode='full')
    lag[i-1] = lags[np.argmax(correlation)]

# specify columns to plot
columns = list(range(0,len(df.columns)))
# set counter
i = 1
values = df.values

# # plot 2d histogram
# # define figure object and size
# plt.figure(figsize=(15,40))
# # plot each column with a for loop
# for variable in columns:
#      plt.subplot(int(len(columns)/2)+1, 2, i)
#      plt.hist2d(y,values[:, variable], bins=(100,100), cmap='Blues')
#      plt.title(df.columns[variable], y=0.5, loc='right')
#      plt.grid()
#      i += 1
# plt.savefig('data/hist2d_plots_d.png')
# plt.close()

i=1
# scatter plots of label vs features
# define figure object and size
plt.figure(figsize=(15,40))
# plot each column with a for loop
for variable in columns:
     plt.subplot(int(len(columns)/2)+1, 2, i)
     plt.scatter(y,values[:, variable],alpha=0.2)
     plt.title(df.columns[variable], y=0.5, loc='right')
     plt.grid()
     i += 1
plt.savefig('data/scatter_plots.png')
plt.close()

i=1
# line plots of label and features
# define figure object and size
plt.figure(figsize=(15,40))
# plot each column with a for loop
for variable in columns:
     plt.subplot(len(columns), 1, i)
     plt.plot(values[:, variable])
     plt.title(df.columns[variable], y=0.5, loc='right')
     i += 1
plt.savefig('data/line_plots.png')
plt.close()

# violing plot of data
# calculate dataset mean and standard deviation
mean = df.mean()
std = df.std()
# normalise dataset with previously calculated values
df_std = (df - mean) / std
# create violin plot
df_std = df_std.melt(var_name='Column', value_name='Normalised')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalised', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
plt.savefig('data/violin.png')
plt.close()

breakpoint()
