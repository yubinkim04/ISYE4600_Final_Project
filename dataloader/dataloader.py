import pandas as pd
import numpy as np

def daigtv2_loader(your_folder):
    data_path = your_folder + "train_v2_drcat_02.csv"
    data = pd.read_csv(data_path)
    data = data.drop_duplicates(subset=["text"])
    #data = data.to_numpy()
    return data