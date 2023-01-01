import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('data.csv')
df.head()

df = df[df['Subject Code'] == 16]
df_1 = df.drop(columns = ['College Code', 'Roll', 'Roll no.', 'Subject Code'])
df_1['Gender'] = df_1['Gender'].replace("Female", "F")
df_1['Gender'] = df_1['Gender'].replace("Male", 'M')
df_1.head()

df_1.isnull().sum()

df_1.dropna(inplace=True)

df_1.isnull().sum()

df_1.head()

enc = LabelEncoder()
enc.fit(['M', 'F'])
enc.classes_
df_1['Gender'] = enc.transform(df_1['Gender'])
df_1.head()

X = df_1[['1st', '2nd', '3rd', '4th','Gender']]
y = df_1[['5th']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)
model.score(X_test, y_test)

y_pred = model.predict(X_test)
plt.plot(range(len(y_test)), y_test, color = 'b')
plt.plot(range(len(y_pred)), y_pred, color = 'g')
# plt.legend()
  
plt.show()

from sklearn.metrics import mean_absolute_error,mean_squared_error
  
mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
mse = mean_squared_error(y_true=y_test,y_pred=y_pred)
rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)
  
print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)

def getInput():
    sem1 = float(input("Enter your 1 Semester CGPA: "))
    sem2 = float(input("Enter your 2 Semester CGPA: "))
    sem3 = float(input("Enter your 3 Semester CGPA: "))
    sem4 = float(input("Enter your 4 Semester CGPA: "))
    return sem1, sem2, sem3, sem4

# import ipywidgets as widgets
sem1, sem2, sem3, sem4 = getInput()
# gender = widgets.Dropdown(
#             options = [("Male", 1), ("Female", 0)],
#             description = "Gender: "
#         )
# gender
g = int(input("Enter 1 - 'Male' or 0 - 'Female': "))
res = model.predict([[sem1, sem2, sem3, sem4, g]])
print(f"Our Prediction of your 5th Semester is {res[0][0]}")


