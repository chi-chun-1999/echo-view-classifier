#%%
# This file is extracting 2 to 3 frame from video

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import os, sys
#%%
df = pd.read_csv('./data/pred/pred_EcoDel3.csv')

#%%

for i in df.iterrows():
     
    file_path = i[1]['file_path']
    file_name = i[1]['file_name']
    file_name = file_name.split('.')[0]
    research_id = i[1]['research_id']
    time = i[1]['time']
    
    np_data = np.load(file_path)
    np_data_length = len(np_data)

    # generate two different random number from 0 to np_data_length

    random_number_1 = random.randint(0,np_data_length-1)
    
    while(True):
        random_number_2 = random.randint(0,np_data_length-1)
        if random_number_1 != random_number_2:
            break
    
    print(random_number_1, random_number_2)
    
    first_img = Image.fromarray(np_data[random_number_1,...])
    second_img = Image.fromarray(np_data[random_number_2,...])
    
    # create directory
    store_dir = '/mnt/chi-chun/data1t/view_classify/EC_strain_del3'+'/'+research_id+'/'+str(time)
    
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
        print(store_dir)
    first_img.save(store_dir+'/'+file_name+'_'+str(random_number_1)+'.png')
    second_img.save(store_dir+'/'+file_name+'_'+str(random_number_2)+'.png')

# %%

for i in df.iterrows():
     
    file_path = i[1]['file_path']
    file_name = i[1]['file_name']
    file_name = file_name.split('.')[0]
    research_id = i[1]['research_id']
    time = i[1]['time']
    
    
    # create directory
    store_dir = '/mnt/chi-chun/data1t/view_classify/EC_strain_del3'+'/'+research_id+'/'+str(time)
    
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
        print(store_dir)
    
    a2c_dir = store_dir+'/a2c'
    a3c_dir = store_dir+'/a3c'
    a4c_dir = store_dir+'/a4c'
    color_dir = store_dir+'/color'
    other_dir = store_dir+'/other'
    
    if not os.path.exists(a2c_dir):
        os.makedirs(a2c_dir)
    if not os.path.exists(a3c_dir):
        os.makedirs(a3c_dir)
    if not os.path.exists(a4c_dir):
        os.makedirs(a4c_dir)
    if not os.path.exists(color_dir):
        os.makedirs(color_dir)
    if not os.path.exists(other_dir):
        os.makedirs(other_dir)
    
#%%
