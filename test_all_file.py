#%%
import glob
import os
import sys
from pandas import DataFrame

#%%
# read all npy files in the /mnt/chi-chun/kmu_data/Echo_strain_del1/*/*/whole_npy/*.npy
# and write the file name into a dataframe
# and save the dataframe as a csv file

# path = '/mnt/chi-chun/kmu_data/Echo_strain_del1/*/*/whole_npy/*.npy'
path = '/mnt/chi-chun/data1t/Echo_strain_del3/*/*/whole_npy/*.npy'
files = glob.glob(path)
# print(files)
# print(len(files))

# write the file name into a dataframe
df = DataFrame(files, columns=['file_path'])
print(df)
#%%
# split file path and file name
_,df['research_id'],df['time'],_, df['file_name'] = df['file_path'].str.rsplit('/', 4).str
print(df)



# %%
# save the dataframe as a csv file
df.to_csv('./Echo_strain_del3.csv', index=False)
# %%

