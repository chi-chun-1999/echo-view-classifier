# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%

df = pd.read_csv('./data/pred/pred_EcoDel2.csv')

print(df)

a2c_view = df[df['pred_label']=='a3c']



# %%
view_np = np.load(a2c_view['file_path'].values[5])
# %%
plt.imshow(view_np[0,...])
plt.show()

# %%
