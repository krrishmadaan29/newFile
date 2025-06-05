from sklearn.datasets import load_iris
import pandas as pd
import os

iris = load_iris(as_frame=True)
df = pd.concat([iris.data, pd.Series(iris.target, name='target')],axis=1)

df.to_csv("data/iris.csv",index=False)
print("iris.csv has been created successfully")