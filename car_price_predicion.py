import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor 
from sklearn import metrics
import joblib

data=pd.read_csv("D:\\ML_projects\\car_price_prediction\\car_data.csv")

print("First 5 rows of dataset:")
print(data.head())

print("Bottom 5 rows of dataset:")
print(data.tail())

print("Shape of dataset:")
print(data.shape)

print("Columns in dataset:",data.shape[1])
print("Rows in dataset:",data.shape[0])

print("Information about dataset:")
print(data.info())

print("Null values in each column:")
print(data.isnull().sum())

print("Statistics summary of dataset:")
print(data.describe())

#///DATA Preprocessing///#
print(data.head(1))

date_time=dt.datetime.now()
print(date_time)

data['Age']=date_time.year-data['Year']
print(data.head(1))

data.drop(['Year'],axis=1,inplace=True)
print(data.head(1))

#///Removing OUTLIERS///#
sns.boxplot(data['Selling_Price'],orient='h')
plt.title("Boxplot of Selling Price")
plt.show()

sorted_prices=sorted(data['Selling_Price'],reverse=True)
print("Sorted Selling Prices:",sorted_prices)

data=data[~(data['Selling_Price']>=33.0) & (data['Selling_Price']<=35.0)]

sorted_prices=sorted(data['Selling_Price'],reverse=True)
print("Sorted Selling Prices after removing outliers:",sorted_prices)

print(data.head(1))

#/// Encoding the Categorical Columns///#
print(data['Fuel_Type'].unique())
data['Fuel_Type']=data['Fuel_Type'].map({'Petrol':0,'Diesel':1,'CNG':2})
print(data['Fuel_Type'].unique())

print(data['Seller_Type'].unique())
data['Seller_Type']=data['Seller_Type'].map({'Dealer':0,'Individual':1})
print(data['Seller_Type'].unique())

print(data['Transmission'].unique())
data['Transmission']=data['Transmission'].map({'Manual':0,'Automatic':1})
print(data['Transmission'].unique())

X=data.drop(['Car_Name','Selling_Price'],axis=1)
y=data['Selling_Price']

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr=LinearRegression()
lr.fit(X_train,y_train)

rf=RandomForestRegressor()
rf.fit(X_train,y_train)

xgb=GradientBoostingRegressor()
xgb.fit(X_train,y_train)

xg=XGBRegressor()
xg.fit(X_train,y_train)

y_pred_lr=lr.predict(X_test)
y_pred_rf=rf.predict(X_test)
y_pred_xgb=xgb.predict(X_test)
y_pred_xg=xg.predict(X_test)

score_lr=metrics.r2_score(y_test,y_pred_lr)
score_rf=metrics.r2_score(y_test,y_pred_rf)
score_xgb=metrics.r2_score(y_test,y_pred_xgb)
score_xg=metrics.r2_score(y_test,y_pred_xg)

final_data=pd.DataFrame({'Models':['Linear Rgression','Random Forest','Gradient Boosting','XGBoost'],
                         'R2_Score':[score_lr,score_rf,score_xgb,score_xg]})
print(final_data)

sns.barplot(x=final_data['Models'],y=final_data['R2_Score'])
plt.title("Model Performance Comparison")
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.show()

#///Save The Model///#

xg=XGBRegressor()
xg_final=xg.fit(X,y)

joblib.dump(xg_final,'car_price_predictor')
model = joblib.load('car_price_predictor')

data_new = pd.DataFrame({
    'Present_Price':5.59,
    'Kms_Driven':27000,
    'Fuel_Type':0,
    'Seller_Type':0,
    'Transmission':0,
    'Owner':2,
    'Age':12
},index=[0])

print(model.predict(data_new))

xg_final.save_model('xgb_model.json')