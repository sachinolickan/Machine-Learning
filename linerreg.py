from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x=[[1],[2],[3],[4],[5],[6]]
y=[[2],[4],[6],[8],[10],[12]]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30)

model=LinearRegression()
model.fit(x_train,y_train)
z=[[20],[15]]
z_predict=model.predict(z)
print(z_predict)



# print(x_test)
# y_predict=model.predict(x_test)
# print(y_predict)

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)