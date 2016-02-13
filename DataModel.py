import pandas as pa
import numpy as np

# load data => {features, label}
def loadData(fileName):
    # Load csv into dataframe
    df = pa.read_csv(fileName)
    
    if "label" not in df.columns:
        df["label"] = None

    pixelColumns = [c for c in df.columns if c != "label"]
    x = np.array(df[pixelColumns])
    y = np.array(df["label"])

    return (x, y)
