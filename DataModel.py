import pandas as pa
import numpy as np
from sklearn import cross_validation
from sklearn import preprocessing

# load data => {features, label}
def loadData(fileName):
    # Load csv into dataframe
    df = pa.read_csv(fileName)
    
    if "label" not in df.columns:
        df["label"] = None

    pixelColumns = [c for c in df.columns if c != "label"]
    x = np.array(df[pixelColumns], dtype=float)
    y = np.array(df["label"], dtype=float)

    x = preprocessing.scale(x, copy = False)

    return (x, y)

def splitData(x, y):
    return cross_validation.train_test_split(x, y, test_size=0.4)

