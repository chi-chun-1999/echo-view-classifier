#%%
import pandas as pd
import numpy as np
from load_data import DataGenerator
from tensorflow.keras.models import load_model
#%%

df = pd.read_csv('./Echo_strain_del2.csv')

# print(df)
print(df['file_path'][0])

#%%

data = np.load(df['file_path'][0])
print(data.shape)


# %%
dg = DataGenerator(df['file_path'],df['file_name'],batch_size=64,num_workers=32)
#%%
# print(dg[0][1])
# %timeit dg[0][1]


# %%

model = load_model('./model/mymodel_echocv_500-500-8_adam_16_0.9394.h5')

pred = model.predict_generator(dg)
#%%
np.save('./data/pred/pred_EcoDel2.npy', pred)

# %%
pred_label = np.argmax(pred,axis=1)
tmp_sum = np.sum(pred>0.5,axis=1)
pred_label[tmp_sum==0]=-1
labels = { 'plax':0, 'psax-av':1, 'psax-mv':2, 'psax-ap':3, 'a4c':4, 'a5c':5, 'a3c':6, 'a2c':7, 'unknown':-1 }
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in pred_label]

df['pred_label'] = predictions
df.to_csv('./data/pred/pred_EcoDel2.csv',index=False)

# %%

df = pd.read_csv('./Echo_strain_del3.csv')

# print(df)
print(df['file_path'][0])

#%%
data = np.load(df['file_path'][0])
print(data.shape)


# %%
dg = DataGenerator(df['file_path'],df['file_name'],batch_size=64,num_workers=32)
#%%
# print(dg[0][1])
# %timeit dg[0][1]


# %%
model = load_model('./model/mymodel_echocv_500-500-8_adam_16_0.9394.h5')

pred = model.predict_generator(dg)
#%%
np.save('./data/pred/pred_EcoDel3.npy', pred)

# %%
pred_label = np.argmax(pred,axis=1)
tmp_sum = np.sum(pred>0.5,axis=1)
pred_label[tmp_sum==0]=-1
labels = { 'plax':0, 'psax-av':1, 'psax-mv':2, 'psax-ap':3, 'a4c':4, 'a5c':5, 'a3c':6, 'a2c':7, 'unknown':-1 }
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in pred_label]

df['pred_label'] = predictions
df.to_csv('./data/pred/pred_EcoDel3.csv',index=False)



