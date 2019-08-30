# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 21:48:18 2019

@author: Jesus kid
"""

import os
os.chdir(r'C:\Users\Jesus kid\Desktop\ML')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

names=('Wife age','Wife education','Husband education','Number of children',
      'Wife religion','wife work?','Husband occupation','Standard-of-living index',
      'Media exposure', 'Contraceptive method')
df = pd.read_csv('cmc.data', sep = ',', header = None, names=names)

df.isnull().sum()

df= pd.get_dummies(df,columns=['Wife education','Husband education', 'Wife religion',
                                     'wife work?','Husband occupation','Standard-of-living index',
      'Media exposure'], drop_first=True )

X=df.drop('Contraceptive method', axis=1)
y=df['Contraceptive method']



'''preprocessing- StandardScaling'''
from sklearn.preprocessing import StandardScaler #standardizing =normalization
sc = StandardScaler()
X_scaled = pd.DataFrame(sc.fit_transform(X[['Wife age', 'Number of children']]), columns=X[['Wife age', 'Number of children']].columns)

X[['Wife age', 'Number of children']]=X_scaled[['Wife age', 'Number of children']]


X2=X.values
y2=y.values


''' get the number of variables that gives u max accuracy score,
 default selector.score in logistic regression = accuracy'''
 
max_accu = 0
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
estimator = LogisticRegression() #use regression model for regression problem
for i in range(1,len(X2[0])+1):
    selector = RFE(estimator, i, step=1)
    selector = selector.fit(X2, y2)
    accuracy = selector.score(X2, y2)
    if max_accu < accuracy:
        sel_features = selector.support_ # this code gives u boolian result
        max_accu = accuracy
        
        
''' get the exact features to use'''        
X2 = X[:,sel_features] 
X2 = pd.DataFrame(X2, columns=('Wife age', 'Number of children', 'Wife education_3', 'Wife education_4', 
                             'Husband education_2', 'Husband education_3', 'Husband education_4', 
                             'Media exposure_1'))
X=X.astype(float)

'''split data'''
from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y,test_size=1/3,random_state=0) #split data for logreg

 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0) #split data for rest of the models

 





'''---------------------------------------------------------------------------------------------'''
'''fit logistic regression to the model'''
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X2_train, y2_train)
y2_pred = logreg.predict(X2_test)



''' score report and confusion matrix'''
from sklearn.model_selection import KFold, cross_val_score
kf= KFold(n_splits=5, shuffle= True, random_state=0)
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score

accuracy_score= accuracy_score(y_test,y2_pred)
macro_precision_score=precision_score(y_test, y2_pred, average='macro')
macro_recall_score=recall_score(y_test, y2_pred, average='macro')
print(accuracy_score, macro_precision_score, macro_recall_score)


cm1=confusion_matrix(y2_test, y2_pred)
cm1=pd.DataFrame(cm1)
cm1.rename(index={0:'No-use',1:"Long-term", 2:'Short-term'}, columns={0:'No-use',1:"Long-term", 2:'Short-term'}, inplace=True)
print(cm1)



pd.value_counts(y_test)

logreg= LogisticRegression()


''' cross validation on accuracy score,  macro precision,  macro recall (test data)'''
accuracy=cross_val_score(logreg.fit(X2_train, y2_train), X2_test,y2_test, cv=kf, scoring='accuracy').mean()
print(accuracy)
precision_macro=cross_val_score(logreg.fit(X2_train, y2_train), X2_test,y2_test, cv=kf, scoring='precision_macro').mean()
print(precision_macro)
recall_macro=cross_val_score(logreg.fit(X2_train, y2_train), X2_test,y2_test, cv=kf, scoring='recall_macro').mean()
print(recall_macro)


''' cross validation on accuracy score,  macro precision,  macro recall (full data)'''
accuracyb=cross_val_score(logreg,X,y, cv=kf, scoring='accuracy').mean()
print(accuracyb)
precision_macrob=cross_val_score(logreg,X,y, cv=kf, scoring='precision_macro').mean()
print(precision_macrob)
recall_macrob=cross_val_score(logreg,X,y, cv=kf, scoring='recall_macro').mean()
print(recall_macrob)









'''------------------------------------------------------------------------------------------------------'''

'''KNeighborsClassifier'''
from sklearn.neighbors import KNeighborsClassifier #KNeighborsRegressor if linear regression
knn1 = KNeighborsClassifier(n_neighbors= 15, p=1, weights='uniform') 


'''find the optimal parameters in KNN'''
param_dict = {
                'n_neighbors': [5,10,15],
                'weights': ['uniform', 'distance' ],
                'p' :[1, 2]          
             }

from sklearn.model_selection import GridSearchCV
knn = GridSearchCV(knn1,param_dict)
knn.fit(X_train,y_train)
knn.best_params_ 
knn.best_score_


'''refit knn to the model with optimal parameters'''
knn=KNeighborsClassifier(n_neighbors= 15, p=1, weights='uniform')
knn.fit(X_train,y_train)
#predictions for test
y_pred2 = knn.predict(X_test)


''' score report and confusion matrix'''

from sklearn.model_selection import KFold, cross_val_score
kf= KFold(n_splits=5, shuffle= True, random_state=0)
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score

accuracy_score= accuracy_score(y_test,y_pred2)
macro_precision_score=precision_score(y_test, y_pred2, average='macro')
macro_recall_score=recall_score(y_test, y_pred2, average='macro')
print(accuracy_score, macro_precision_score, macro_recall_score)


cm2=confusion_matrix(y_test, y_pred2)
cm2=pd.DataFrame(cm2)
cm2.rename(index={0:'No-use',1:"Long-term", 2:'Short-term'}, columns={0:'No-use',1:"Long-term", 2:'Short-term'}, inplace=True)
print(cm2)


''' cross validation on accuracy score,  macro precision,  macro recall (test data)'''
accuracy2=cross_val_score(knn.fit(X_train, y_train),X_test,y_test, cv=kf, scoring='accuracy').mean()
print(accuracy2)
precision_macro2=cross_val_score(knn.fit(X_train, y_train),X_test,y_test, cv=kf, scoring='precision_macro').mean()
print(precision_macro2)
recall_macro2=cross_val_score(knn.fit(X_train, y_train),X_test,y_test, cv=kf, scoring='recall_macro').mean()
print(recall_macro2)


''' cross validation on accuracy score,  macro precision,  macro recall (full data)'''
accuracy2b=cross_val_score(knn1,X,y, cv=kf, scoring='accuracy').mean()
print(accuracy2b)
precision_macro2b=cross_val_score(knn1,X,y, cv=kf, scoring='precision_macro').mean()
print(precision_macro2b)
recall_macro2b=cross_val_score(knn1,X,y, cv=kf, scoring='recall_macro').mean()
print(recall_macro2b)










'''------------------------------------------------------------------------------------------------------'''
'''SVC'''

from sklearn.svm import SVC # svc= svm for classification; SVR= svm for regressor
svc1=SVC(C=1, degree=2, kernel='rbf')


'''find the optimal parameters in svc'''
param_dict = {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [2,3,4,5],
                'C' :[0.001, 0.01, 0.1, 1, 10]          
             }


from sklearn.model_selection import GridSearchCV
svc = GridSearchCV(svc1,param_dict)
svc.fit(X_train,y_train)
svc.best_params_ 


'''refit SVC to the model with optimal parameters'''
svc=SVC(C=1, degree=2, kernel='rbf')
svc.fit(X_train,y_train)
y_pred3 = svc.predict(X_test)

''' score report and confusion matrix'''
from sklearn.model_selection import KFold, cross_val_score
kf= KFold(n_splits=5, shuffle= True, random_state=0)
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score

accuracy_score= accuracy_score(y_test,y_pred3)
macro_precision_score=precision_score(y_test, y_pred3, average='macro')
macro_recall_score=recall_score(y_test, y_pred3, average='macro')
print(accuracy_score, macro_precision_score, macro_recall_score)


cm3=confusion_matrix(y_test, y_pred3)
cm3=pd.DataFrame(cm3)
cm3.rename(index={0:'No-use',1:"Long-term", 2:'Short-term'}, columns={0:'No-use',1:"Long-term", 2:'Short-term'}, inplace=True)
print(cm3)

''' cross validation on accuracy score,  macro precision,  macro recall (test data)'''
accuracy3=cross_val_score(svc.fit(X_train,y_train).fit(X_train, y_train),X_test,y_test, cv=kf, scoring='accuracy').mean()
print(accuracy3)
precision_macro3=cross_val_score(svc.fit(X_train, y_train),X_test,y_test, cv=kf, scoring='precision_macro').mean()
print(precision_macro3)
recall_macro3=cross_val_score(svc.fit(X_train, y_train),X_test,y_test, cv=kf, scoring='recall_macro').mean()
print(recall_macro3)


''' cross validation on accuracy score,  macro precision,  macro recall (full data)'''
accuracy3b=cross_val_score(svc1,X,y, cv=kf, scoring='accuracy').mean()
print(accuracy3b)
precision_macro3b=cross_val_score(svc1,X,y, cv=kf, scoring='precision_macro').mean()
print(precision_macro3b)
recall_macro3b=cross_val_score(svc1,X,y, cv=kf, scoring='recall_macro').mean()
print(recall_macro3b)



'''------------------------------------------------------------------------------------------------------'''
'''RandomForestClassifier'''
''' plot, to find the optimal n_estimator(95) and  max_depth(6)'''

from sklearn.ensemble import RandomForestClassifier 
test_scores=[]
for n in range(50,100):
    model=RandomForestClassifier (n_estimators=n, max_depth=10, random_state=0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    test_scores.append(model.score(X_test, y_test))
   
plt.plot(range(50, 100), test_scores)
plt.xlabel('n of DTs')
plt.ylabel('accuracy')


test_scores1=[]
for k in range(1,60):
    model=RandomForestClassifier(n_estimators=95, max_depth=k, random_state=0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    test_scores1.append(model.score(X_test, y_test))
    
plt.plot(range(1, 60), test_scores1)
plt.xlabel('n of depth')
plt.ylabel('accuracy')



'''fit RandomForestClassifier to the model'''
rfc1= RandomForestClassifier(n_estimators=95, max_depth=6)
rfc= RandomForestClassifier(n_estimators=95, max_depth=6)
rfc.fit(X_train,y_train)
y_pred4 = rfc.predict(X_test)


''' score report and confusion matrix'''
from sklearn.model_selection import KFold, cross_val_score
kf= KFold(n_splits=5, shuffle= True, random_state=0)
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score

accuracy_score= accuracy_score(y_test,y_pred4)
macro_precision_score=precision_score(y_test, y_pred4, average='macro')
macro_recall_score=recall_score(y_test, y_pred4, average='macro')
print(accuracy_score, macro_precision_score, macro_recall_score)

cm4=confusion_matrix(y_test, y_pred4)
cm4=pd.DataFrame(cm4)
cm4.rename(index={0:'No-use',1:"Long-term", 2:'Short-term'}, columns={0:'No-use',1:"Long-term", 2:'Short-term'}, inplace=True)
print(cm4)

''' cross validation on accuracy score,  macro precision,  macro recall (test data)'''
accuracy4=cross_val_score(rfc.fit(X_train,y_train).fit(X_train, y_train),X_test,y_test, cv=kf, scoring='accuracy').mean()
print(accuracy4)
precision_macro4=cross_val_score(rfc.fit(X_train, y_train),X_test,y_test, cv=kf, scoring='precision_macro').mean()
print(precision_macro4)
recall_macro4=cross_val_score(rfc.fit(X_train, y_train),X_test,y_test, cv=kf, scoring='recall_macro').mean()
print(recall_macro4)


''' cross validation on accuracy score,  macro precision,  macro recall (full data)'''
accuracy4b=cross_val_score(rfc1,X,y, cv=kf, scoring='accuracy').mean()
print(accuracy4b)
precision_macro4b=cross_val_score(rfc1,X,y, cv=kf, scoring='precision_macro').mean()
print(precision_macro4b)
recall_macro4b=cross_val_score(rfc1,X,y, cv=kf, scoring='recall_macro').mean()
print(recall_macro4b)








'''------------------------------------------------------------------------------------------------------'''
'''logreg- OneVsOneClassifier'''

from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import SGDClassifier # loss=squared.loss= "linear reg"

base_model = SGDClassifier() 

'''find the optimal parameters in SGDRegressor()'''
param_dict = {
                'penalty': ['l1','l2','elasticnet'] ,#regularization
                'alpha':[0.001,0.01,0.1,1],
                        
                'learning_rate': ['optimal','constant','invscaling'],
                'eta0': [0.001,0.01,0.1,1]
             }


from sklearn.model_selection import GridSearchCV
model = GridSearchCV(base_model,param_dict,cv=5)
model.fit(X_train,y_train)
model.best_params_ 


'''fit ovo(SGDRegressor) to the model with optimal parameters'''
base_model = SGDClassifier(alpha= 0.001, eta0=1, learning_rate='invscaling', penalty='l1') 
ovo1 = OneVsOneClassifier(base_model)

ovo = OneVsOneClassifier(base_model)
ovo.fit(X_train,y_train)
y_pred5 = ovo.predict(X_test)


''' score report and confusion matrix'''
from sklearn.model_selection import KFold, cross_val_score
kf= KFold(n_splits=5, shuffle= True, random_state=0)
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score

accuracy_score= accuracy_score(y_test,y_pred5)
macro_precision_score=precision_score(y_test, y_pred5, average='macro')
macro_recall_score=recall_score(y_test, y_pred5, average='macro')
print(accuracy_score, macro_precision_score, macro_recall_score)


cm5=confusion_matrix(y_test, y_pred5)
cm5=pd.DataFrame(cm5)
cm5.rename(index={0:'No-use',1:"Long-term", 2:'Short-term'}, columns={0:'No-use',1:"Long-term", 2:'Short-term'}, inplace=True)
print(cm5)


''' cross validation on accuracy score,  macro precision,  macro recall (test data)'''
accuracy5=cross_val_score(ovo.fit(X_train,y_train).fit(X_train, y_train),X_test,y_test, cv=kf, scoring='accuracy').mean()
print(accuracy5)
precision_macro5=cross_val_score(ovo.fit(X_train, y_train),X_test,y_test, cv=kf, scoring='precision_macro').mean()
print(precision_macro5)
recall_macro5=cross_val_score(ovo.fit(X_train, y_train),X_test,y_test, cv=kf, scoring='recall_macro').mean()
print(recall_macro5)


''' cross validation on accuracy score,  macro precision,  macro recall (full data)'''
accuracy5b=cross_val_score(ovo1,X,y, cv=kf, scoring='accuracy').mean()
print(accuracy5b)
precision_macro5b=cross_val_score(ovo1,X,y, cv=kf, scoring='precision_macro').mean()
print(precision_macro5b)
recall_macro5b=cross_val_score(ovo1,X,y, cv=kf, scoring='recall_macro').mean()
print(recall_macro5b)
