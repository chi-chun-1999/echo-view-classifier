#%%
# This file will get label from directory, and save into csv file
import pandas as pd
import os
import numpy as np
import glob

#%%

root_dir = '/mnt/chi-chun/data1t/view_classify/'

sub_dir = ['EC_strain_del1','EC_strain_del2','EC_strain_del3']

#%%
# path = '/mnt/chi-chun/data1t/view_classify/EC_strain_del1/0/reseach_id/time/label/file_name.png'
df = pd.DataFrame(columns=['file_path','sub_dir','file_name','research_id','time','label'])

file_list = []

for i in sub_dir:
    path = root_dir+i+'/'+str(0)+'/'+'*/*/*/*.png'
    files = glob.glob(path)
    sub_dir_name = i
    
    for j in files:
        print(j)
        tmp = j.split('/')
        file_name = tmp[-1]
        file_name = file_name.split('_')[0]
        label = tmp[-2]
        research_id = tmp[-4]
        time = tmp[-3]
        if file_name not in file_list:
            file_list.append(file_name)
            # df = df.append({'file_path':j,'file_name':file_name,'research_id':research_id,'time':time,'label':label},ignore_index=True)
            
            new_df = pd.DataFrame({'file_path':j,'sub_dir':sub_dir_name,'file_name':file_name,'research_id':research_id,'time':time,'label':label},index=[0])
            
            df = pd.concat([df,new_df],ignore_index=True)
    
print(df)
    

# %%

df.to_csv('./data/echo_label.csv',index=False)

# %%
