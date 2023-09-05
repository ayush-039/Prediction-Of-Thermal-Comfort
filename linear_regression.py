##### csv file reading #####
import pandas as pd
df = pd.read_csv('cleaned_data.csv')
df

##### data spliting  #####
from sklearn.model_selection import train_test_split

##### for indoor temperature #####
X = df[['CCSP','CHSP','OTEMP','ORH','OAV','YEAR','MONTH','DAY']]
y = df['ITEMP']
x_train_temp, x_test_tey_train_temp, y_test_temp = train_test_split(X,y,test_size=0.2,random_state=20)

##### for indoor humidity #####
X = df[['CCSP','CHSP','OTEMP','ORH','OAV','YEAR','MONTH','DAY']]
y = df['IRH']
x_train_rh, x_test_rh, y_train_rh, y_test_rh = train_test_split(X,y,test_size=0.2,random_state=20)

##### for indoor humidity #####
X = df[['CCSP','CHSP','OTEMP','ORH','OAV','YEAR','MONTH','DAY']]
y = df['IMRT']
x_train_mrt, x_test_mrt, y_train_mrt, y_test_mrt = train_test_split(X,y,test_size=0.2,random_state=20)

##### coarse gained #####
X_ = df[['CCSP','CHSP','OTEMP','ORH','OAV','YEAR','MONTH','DAY']]
y_ = df['PMV']
x_train_, x_test_, y_train_, y_test_ = train_test_split(X_,y_,test_size=0.2,random_state=20)

####### standard scaler #######
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_=scaler.fit_transform(x_train_)
x_test_ = scaler.transform(x_test_)


##### Linear Regression training #####
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train_,y_train_)
predictions = lr.predict(x_test_)
print("training score")
print(lr.score(x_train_,y_train_))
print("testing score")
print(lr.score(x_test_,y_test_))

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
mae = mean_absolute_error(y_test_,predictions)
mse = mean_squared_error(y_test_,predictions)
print(f"mae: {mae}")
print(f"mse: {mse}")

##### graph ploting #####
