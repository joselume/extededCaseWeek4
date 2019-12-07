#importing all the required ML packages
import pandas as pd 
import pickle

from sklearn.svm import SVC #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.ensemble import VotingClassifier  

import warnings
warnings.filterwarnings('ignore')

#loading the data
df = pd.read_csv('data.csv')

# Drop useless variables
df = df.drop(['Unnamed: 32','id'],axis = 1)

# Reassign target
df.diagnosis.replace(to_replace = dict(M = 1, B = 0), inplace = True)

# Split data
X = df[['concave points_mean', 'texture_mean', 'area_mean', 'compactness_mean']]
y = df['diagnosis']
kfold = StratifiedKFold(n_splits=5, shuffle = True ,random_state=2019)
for train, test in kfold.split(X, y):
        print('=)')

# Best random forest 
best_RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=None,
                                 max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0,
                                 min_impurity_split=None,min_samples_leaf=1, min_samples_split=2,
                                 min_weight_fraction_leaf=0.0, n_estimators=550, n_jobs=1,oob_score=False,
                                 random_state=0, verbose=0, warm_start=False)
# Best Suport Vector Machine
best_SVM = SVC(C=0.9, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr',
               degree=3, gamma=0.1, kernel='linear',max_iter=-1, probability=True, random_state=None,
               shrinking=True,tol=0.001, verbose=False)

# Enemble Model
ensemble_lin_rbf = VotingClassifier(estimators=[('RFor', best_RF),
                                               ('NB', GaussianNB()),
                                               ('svm', best_SVM)], 
                       voting='soft').fit(X.iloc[train],y.iloc[train])

print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(X.iloc[test],y.iloc[test]))

cross=cross_val_score(ensemble_lin_rbf,X,y, cv = 5,scoring = "roc_auc")
print('The cross validated score is',cross.mean())


pickle.dump(ensemble_lin_rbf, open('model_aws.pkl', 'wb'))
