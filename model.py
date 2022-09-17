
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
hd=pd.read_csv(r'heart (1).csv')
print(hd.head())

print(hd.shape)
hd.info()
print(hd.describe())
hd.isnull().sum()
hd['target'].value_counts()
x=hd.drop(columns='target',axis=1)
print(x)

hd.nunique(axis=0)

x1=hd.target.unique()
y1=hd['target'].value_counts()

sns.barplot(x=x1,y=y1)

sns.scatterplot(x='thalach',y='age',data=hd,hue='sex')
plt.show()

x_train, x_test,y_train, y_test= train_test_split(x,y,test_size=0.3,stratify=y, random_state=9)

print("shape of x, training-x and testing-x")
print(x.shape,x_train.shape,x_test.shape)
print("shape of y, training-y and testing-y")
print(y.shape,y_train.shape,y_test.shape)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

lr=LogisticRegression()
lr.fit(x_train,y_train)

hd_lr_train_predict=lr.predict(x_train)
hd_lr_test_predict=lr.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse=mean_squared_error(y_train,hd_lr_train_predict)
lr_train_r2=r2_score(y_train,hd_lr_train_predict)

lr_test_mse=mean_squared_error(y_test,hd_lr_test_predict)
lr_test_r2=r2_score(y_test,hd_lr_test_predict)

lr_test_clf=classification_report(y_test,hd_lr_test_predict)
lr_test_cnfm=confusion_matrix(y_test, hd_lr_test_predict)


lr_results=pd.DataFrame(['Logistic Regression',lr_train_mse,lr_test_mse,lr_train_r2,lr_test_r2]).transpose()
lr_results.columns=['Method',' Training MSE ',' Training R2 ','Test MSE','Test R2']
print(lr_results)

print("_____________________________________________________________________________________")
print("CLASSIFICATION REPORT")
print(lr_test_clf)
print("_____________________________________________________________________________________")
print("CONFUSION MATRIX")
print(lr_test_cnfm)

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(max_depth=2,random_state=42)
rf.fit(x_train, y_train) #data is trained using rf.fit() function

hd_rf_train_predict=rf.predict(x_train)
hd_rf_test_predict=rf.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score

rf_train_mse=mean_squared_error(y_train,hd_rf_train_predict)
rf_train_r2=r2_score(y_train,hd_rf_train_predict)

rf_test_mse=mean_squared_error(y_test,hd_rf_test_predict)
rf_test_r2=r2_score(y_test,hd_rf_test_predict)

rf_test_clf=classification_report(y_test,hd_rf_test_predict)
rf_test_cnfm=confusion_matrix(y_test, hd_rf_test_predict)

rf_results=pd.DataFrame(['Random Forest',rf_train_mse,rf_test_mse,rf_train_r2,rf_test_r2]).transpose()
print(rf_results.columns=['Method',' Training MSE ',' Training R2 ','Test MSE','Test R2'])
print(rf_results)

print("_____________________________________________________________________________________")
print("CLASSIFICATION REPORT")
print(rf_test_clf)
print("_____________________________________________________________________________________")
print("CONFUSION MATRIX")
print(rf_test_cnfm)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier( random_state=42)
dt.fit(x_train,y_train)

hd_dt_train_predict=dt.predict(x_train)
hd_dt_test_predict=dt.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score,classification_report,confusion_matrix

dt_train_mse=mean_squared_error(y_train,hd_dt_train_predict)
dt_train_r2=r2_score(y_train,hd_dt_train_predict)

dt_train_clf=classification_report(y_train, hd_dt_train_predict)
dt_train_cnfm=confusion_matrix(y_train, hd_dt_train_predict)

dt_test_mse=mean_squared_error(y_test,hd_dt_test_predict)
dt_test_r2=r2_score(y_test,hd_dt_test_predict)


dt_test_clf=classification_report(y_test,hd_dt_test_predict)
dt_test_cnfm=confusion_matrix(y_test, hd_dt_test_predict)

dt_results=pd.DataFrame(['Decision tree',dt_train_mse,dt_test_mse,dt_train_r2,dt_test_r2]).transpose()
dt_results.columns=['Method',' Training MSE ',' Training R2 ','Test MSE','Test R2']

print(dt_results)

print("_____________________________________________________________________________________")
print("CLASSIFICATION REPORT")
print(dt_test_clf)
print("_____________________________________________________________________________________")
print("CONFUSION MATRIX")
print(dt_test_cnfm)

from sklearn.svm import SVC
svm = SVC(kernel='rbf', gamma=0.1, C=1.0)
svm.fit(x_train, y_train)
hd_svm_train_predict=svm.predict(x_train)
hd_svm_test_predict=svm.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
svm_train_mse=mean_squared_error(y_train,hd_svm_train_predict)
svm_train_r2=r2_score(y_train,hd_svm_train_predict)

svm_test_mse=mean_squared_error(y_test,hd_svm_test_predict)
svm_test_r2=r2_score(y_test,hd_svm_test_predict)

svm_test_clf=classification_report(y_test,hd_svm_test_predict)
svm_test_cnfm=confusion_matrix(y_test, hd_svm_test_predict)
svm_results=pd.DataFrame(['Support vector Machine',svm_train_mse,svm_test_mse,svm_train_r2,svm_test_r2]).transpose()
svm_results.columns=['Method',' Training MSE ',' Training R2 ','Test MSE','Test R2']
print(svm_results)

print("_____________________________________________________________________________________")
print("CLASSIFICATION REPORT")
print(svm_test_clf)
print("_____________________________________________________________________________________")
print("CONFUSION MATRIX")
print(svm_test_cnfm)

