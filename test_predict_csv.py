#%%
# This file is for checking the research_id that does not have a2c, a3c, a4c label
import pandas as pd

df = pd.read_csv('./data/pred/pred_EcoDel1.csv')

#%%
unique_research_id = df['research_id'].unique()
# %%

a2c_view = df[df['pred_label']=='a2c']

without_a2c_research_id = []
for r in unique_research_id:
    # search each research_id wether a2c label or not
    # there is no a2c in research_id, print research_id
    if r not in a2c_view['research_id'].values:
        without_a2c_research_id.append(r)
        print(r)

    
# %%
a3c_view = df[df['pred_label']=='a3c']

without_a3c_research_id = []
for r in unique_research_id:
    # search each research_id wether a3c label or not
    # there is no a3c in research_id, print research_id
    if r not in a3c_view['research_id'].values:
        without_a3c_research_id.append(r)
        print(r)
# %%

a4c_view = df[df['pred_label']=='a4c']

without_a4c_research_id = []
for r in unique_research_id:
    # search each research_id wether a4c label or not
    # there is no a4c in research_id, print research_id
    if r not in a4c_view['research_id'].values:
        without_a4c_research_id.append(r)
        print(r)


# %%
