# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:38:43 2019

@author: Saiprasad
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
# Always have X as a matrix and y as a vector
X = dataset.iloc[:,2:4].values
y = dataset.iloc[:,4].values



"""
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
#imputer = imputer.fit(X[:,1:3])
#X[:,1:3] = imputer.transform(X[:,1:3])
X[:,1:3] = imputer.fit_transform(X[:,1:3])
"""

"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
labelencoder_X = labelencoder_X.fit(X[:,0])
X[:,0] = labelencoder_X.transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
labelencoder_y = labelencoder_y.fit(y)
y = labelencoder_y.transform(y)
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25 ,random_state=0)





"""
# fitiing the SLR model in the matrix created 
linreg_2 = LinearRegression()
linreg_2 = linreg_2.fit(X_pol, y)
y_pred_pol = linreg_2.predict(X_pol)
"""
"""
# Better graph 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
"""

# NEED ONLY ARRAYS FOR FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# fitting the model ("regressor" is created here)
# TRY USING VARIOUS OPTIONS FOR THE KERNELS TO SEE WHICH FITS THE DATA BEST
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
classifier.fit(X_train, y_train)





# predicting the output for a particular value ([[]]) is to make sure that the inout for the transform function is a array
# predict and transform operations are used to make sure that the proper scaling is maintained as we are involving feature scaling in the process
#y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

y_pred = classifier.predict(X_test) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# visualising the classifier model
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1 , X2 = np.meshgrid(np.arange(X_set[:,0].min()-1,X_set[:,0].max()+1,0.01),
                      np.arange(X_set[:,1].min()-1,X_set[:,1].max()+1,0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T ).reshape(X1.shape),alpha=0.50,cmap=ListedColormap(('red','green')) );
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(),X2.max())
plt.scatter(X_set[:,0],X_set[:,1])
x3 = list(enumerate(np.unique(y_train)))
for i in range(0,len(y_set)):
    plt.scatter(X_set[i,0],X_set[i,1],c=ListedColormap(('red','green'))(y_set[i]))



# TEST RESULTS

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, classifier.predict(X_test) 
X1 , X2 = np.meshgrid(np.arange(X_set[:,0].min()-1,X_set[:,0].max()+1,0.01),
                      np.arange(X_set[:,1].min()-1,X_set[:,1].max()+1,0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T ).reshape(X1.shape),alpha=0.50,cmap=ListedColormap(('red','green')) );
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(),X2.max())
plt.scatter(X_set[:,0],X_set[:,1])
x3 = list(enumerate(np.unique(y_train)))
for i in range(0,len(y_set)):
    plt.scatter(X_set[i,0],X_set[i,1],c=ListedColormap(('red','green'))(y_set[i]))








