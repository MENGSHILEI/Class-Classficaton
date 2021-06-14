# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
/kaggle/input/glass/glass.csv
df=pd.read_csv("/kaggle/input/glass/glass.csv")
df.shape
(214, 10)
df.head(10)
RI	Na	Mg	Al	Si	K	Ca	Ba	Fe	Type
0	1.52101	13.64	4.49	1.10	71.78	0.06	8.75	0.0	0.00	1
1	1.51761	13.89	3.60	1.36	72.73	0.48	7.83	0.0	0.00	1
2	1.51618	13.53	3.55	1.54	72.99	0.39	7.78	0.0	0.00	1
3	1.51766	13.21	3.69	1.29	72.61	0.57	8.22	0.0	0.00	1
4	1.51742	13.27	3.62	1.24	73.08	0.55	8.07	0.0	0.00	1
5	1.51596	12.79	3.61	1.62	72.97	0.64	8.07	0.0	0.26	1
6	1.51743	13.30	3.60	1.14	73.09	0.58	8.17	0.0	0.00	1
7	1.51756	13.15	3.61	1.05	73.24	0.57	8.24	0.0	0.00	1
8	1.51918	14.04	3.58	1.37	72.08	0.56	8.30	0.0	0.00	1
9	1.51755	13.00	3.60	1.36	72.99	0.57	8.40	0.0	0.11	1
df.describe().T
count	mean	std	min	25%	50%	75%	max
RI	214.0	1.518365	0.003037	1.51115	1.516523	1.51768	1.519157	1.53393
Na	214.0	13.407850	0.816604	10.73000	12.907500	13.30000	13.825000	17.38000
Mg	214.0	2.684533	1.442408	0.00000	2.115000	3.48000	3.600000	4.49000
Al	214.0	1.444907	0.499270	0.29000	1.190000	1.36000	1.630000	3.50000
Si	214.0	72.650935	0.774546	69.81000	72.280000	72.79000	73.087500	75.41000
K	214.0	0.497056	0.652192	0.00000	0.122500	0.55500	0.610000	6.21000
Ca	214.0	8.956963	1.423153	5.43000	8.240000	8.60000	9.172500	16.19000
Ba	214.0	0.175047	0.497219	0.00000	0.000000	0.00000	0.000000	3.15000
Fe	214.0	0.057009	0.097439	0.00000	0.000000	0.00000	0.100000	0.51000
Type	214.0	2.780374	2.103739	1.00000	1.000000	2.00000	3.000000	7.00000
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 214 entries, 0 to 213
Data columns (total 10 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   RI      214 non-null    float64
 1   Na      214 non-null    float64
 2   Mg      214 non-null    float64
 3   Al      214 non-null    float64
 4   Si      214 non-null    float64
 5   K       214 non-null    float64
 6   Ca      214 non-null    float64
 7   Ba      214 non-null    float64
 8   Fe      214 non-null    float64
 9   Type    214 non-null    int64  
dtypes: float64(9), int64(1)
memory usage: 16.8 KB
df.isnull().sum()
RI      0
Na      0
Mg      0
Al      0
Si      0
K       0
Ca      0
Ba      0
Fe      0
Type    0
dtype: int64
import seaborn as sns
p=sns.pairplot(df, hue = 'Type')
/opt/conda/lib/python3.7/site-packages/seaborn/distributions.py:306: UserWarning: Dataset has 0 variance; skipping density estimate.
  warnings.warn(msg, UserWarning)
/opt/conda/lib/python3.7/site-packages/seaborn/distributions.py:306: UserWarning: Dataset has 0 variance; skipping density estimate.
  warnings.warn(msg, UserWarning)
/opt/conda/lib/python3.7/site-packages/seaborn/distributions.py:306: UserWarning: Dataset has 0 variance; skipping density estimate.
  warnings.warn(msg, UserWarning)
  ## checking the balance of the data by plotting the count of type by their value
color_wheel = {1: "#0392cf",2: "#7bc043"}
colors = df["Type"].map(lambda x: color_wheel.get(x + 1))
print(df.Type.value_counts())
p=df['Type'].value_counts().plot(kind="bar")
2    76
1    70
7    29
3    17
5    13
6     9
Name: Type, dtype: int64

#dividing data
X = df.drop("Type",axis = 1)
y = df.Type
X.head(10)
RI	Na	Mg	Al	Si	K	Ca	Ba	Fe
0	1.52101	13.64	4.49	1.10	71.78	0.06	8.75	0.0	0.00
1	1.51761	13.89	3.60	1.36	72.73	0.48	7.83	0.0	0.00
2	1.51618	13.53	3.55	1.54	72.99	0.39	7.78	0.0	0.00
3	1.51766	13.21	3.69	1.29	72.61	0.57	8.22	0.0	0.00
4	1.51742	13.27	3.62	1.24	73.08	0.55	8.07	0.0	0.00
5	1.51596	12.79	3.61	1.62	72.97	0.64	8.07	0.0	0.26
6	1.51743	13.30	3.60	1.14	73.09	0.58	8.17	0.0	0.00
7	1.51756	13.15	3.61	1.05	73.24	0.57	8.24	0.0	0.00
8	1.51918	14.04	3.58	1.37	72.08	0.56	8.30	0.0	0.00
9	1.51755	13.00	3.60	1.36	72.99	0.57	8.40	0.0	0.11
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#scalling all data to be with the same scale 
scaler = StandardScaler()
X =scaler.fit_transform(X)
X
array([[ 0.87286765,  0.28495326,  1.25463857, ..., -0.14576634,
        -0.35287683, -0.5864509 ],
       [-0.24933347,  0.59181718,  0.63616803, ..., -0.79373376,
        -0.35287683, -0.5864509 ],
       [-0.72131806,  0.14993314,  0.60142249, ..., -0.82894938,
        -0.35287683, -0.5864509 ],
       ...,
       [ 0.75404635,  1.16872135, -1.86551055, ..., -0.36410319,
         2.95320036, -0.5864509 ],
       [-0.61239854,  1.19327046, -1.86551055, ..., -0.33593069,
         2.81208731, -0.5864509 ],
       [-0.41436305,  1.00915211, -1.86551055, ..., -0.23732695,
         3.01367739, -0.5864509 ]])
#importing train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42,shuffle=True, stratify=y)
print('X_train shape : ', X_train.shape )
print('y_train shape : ', y_train.shape )
print('X_test shape : ', X_test.shape )
print('y_test shape : ', y_test.shape )
X_train shape :  (143, 9)
y_train shape :  (143,)
X_test shape :  (71, 9)
y_test shape :  (71,)
K Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier


test_scores = []
train_scores = []
K=[]
for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))
    K.append(i)
chosing the best prameters
results= { 'train_scores':train_scores,
         'test_scores':test_scores,
         'K':K}
resultsdf =  pd.DataFrame(results,
        columns=['train_scores','test_scores', 'K'])
resultsdf
train_scores	test_scores	K
0	1.000000	0.788732	1
1	0.783217	0.760563	2
2	0.797203	0.690141	3
3	0.713287	0.676056	4
4	0.720280	0.633803	5
5	0.699301	0.661972	6
6	0.685315	0.676056	7
7	0.692308	0.690141	8
8	0.664336	0.676056	9
9	0.671329	0.661972	10
10	0.671329	0.676056	11
11	0.664336	0.633803	12
12	0.671329	0.633803	13
13	0.657343	0.690141	14
print('train_scores  are :' , train_scores )
print('test_scores  are :' , test_scores )
print('K  are :' , K )
train_scores  are : [1.0, 0.7832167832167832, 0.7972027972027972, 0.7132867132867133, 0.7202797202797203, 0.6993006993006993, 0.6853146853146853, 0.6923076923076923, 0.6643356643356644, 0.6713286713286714, 0.6713286713286714, 0.6643356643356644, 0.6713286713286714, 0.6573426573426573]
test_scores  are : [0.7887323943661971, 0.7605633802816901, 0.6901408450704225, 0.676056338028169, 0.6338028169014085, 0.6619718309859155, 0.676056338028169, 0.6901408450704225, 0.676056338028169, 0.6619718309859155, 0.676056338028169, 0.6338028169014085, 0.6338028169014085, 0.6901408450704225]
K  are : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
print('max_train score', max(train_scores))
print('min train score', min(train_scores))
max_train score 1.0
min train score 0.6573426573426573
print('max test score', max(test_scores))
print('max test score', min(test_scores))
max test score 0.7887323943661971
max test score 0.6338028169014085
import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import confusion_matrix
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
K Neighbors Classifier
KNNModel = KNeighborsClassifier(1)
KNNModel.fit(X_train,y_train)

train_score = KNNModel.score(X_train,y_train)
test_score = KNNModel.score(X_test,y_test)
print( 'train_score ', train_score)
print( 'test_score ',test_score )
train_score  1.0
test_score  0.7887323943661971
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-24-ce66baa38ac5> in <module>
      1 from sklearn import metrics
      2 # Model Accuracy, how often is the classifier correct?
----> 3 print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

NameError: name 'y_pred' is not defined
confusion_matrix
#prediction step
y_pred = knn.predict(X_test)


confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
Predicted	1	2	5	7	All
True					
1	20	3	0	0	23
2	6	19	0	0	25
3	5	1	0	0	6
5	0	2	2	0	4
6	2	1	0	0	3
7	1	1	0	8	10
All	34	27	2	8	71
classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
              precision    recall  f1-score   support

           1       0.59      0.87      0.70        23
           2       0.70      0.76      0.73        25
           3       0.00      0.00      0.00         6
           5       1.00      0.50      0.67         4
           6       0.00      0.00      0.00         3
           7       1.00      0.80      0.89        10

    accuracy                           0.69        71
   macro avg       0.55      0.49      0.50        71
weighted avg       0.64      0.69      0.65        71

Accuracy
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
Accuracy: 0.6901408450704225
Random Forest Model
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier


#Train the model using the training sets y_pred=clf.predict(X_test)
RFclfModel.fit(X_train,y_train)

y_pred_RFclfModel=RFclfModel.predict(X_test)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-28-89d77670e97f> in <module>
      4 
      5 #Train the model using the training sets y_pred=clf.predict(X_test)
----> 6 RFclfModel.fit(X_train,y_train)
      7 
      8 y_pred_RFclfModel=RFclfModel.predict(X_test)

NameError: name 'RFclfModel' is not defined
Accuracy
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_RFclfModel))
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-29-f12b5e0c636d> in <module>
      1 from sklearn import metrics
      2 # Model Accuracy, how often is the classifier correct?
----> 3 print("Accuracy:",metrics.accuracy_score(y_test, y_pred_RFclfModel))

NameError: name 'y_pred_RFclfModel' is not defined
confusion_matrix
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred_RFclfModel, rownames=['True'], colnames=['Predicted'], margins=True)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-30-7b67a7915c58> in <module>
      1 confusion_matrix(y_test,y_pred)
----> 2 pd.crosstab(y_test, y_pred_RFclfModel, rownames=['True'], colnames=['Predicted'], margins=True)

NameError: name 'y_pred_RFclfModel' is not defined
classification_report
print(classification_report(y_test, y_pred_RFclfModel))
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-31-ba8e50fb4cc3> in <module>
----> 1 print(classification_report(y_test, y_pred_RFclfModel))

NameError: name 'y_pred_RFclfModel' is not defined
df.tail()
RI	Na	Mg	Al	Si	K	Ca	Ba	Fe	Type
209	1.51623	14.14	0.0	2.88	72.61	0.08	9.18	1.06	0.0	7
210	1.51685	14.92	0.0	1.99	73.06	0.00	8.40	1.59	0.0	7
211	1.52065	14.36	0.0	2.02	73.42	0.00	8.44	1.64	0.0	7
212	1.51651	14.38	0.0	1.94	73.61	0.00	8.48	1.57	0.0	7
213	1.51711	14.23	0.0	2.08	73.36	0.00	8.62	1.67	0.0	7
sns.pairplot(df [["RI", "Na", "Mg", "Type"]], diag_kind="kde");

df.to_csv("result3.csv")
