from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

data=pd.read_csv('sample.csv')
data=data.dropna()



x=data.iloc[:,0:1].values
y=data.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30)

model=LinearRegression()
model.fit(x_train,y_train)
z=[[1400],[1500],[1350]]
z_predict=model.predict(z)
print(z_predict)