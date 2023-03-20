import numpy as np
import pandas as pd
import seaborn as sns
data= pd.read_excel('./drive/My Drive/Reserach Work/MG Sir/Pritam/Database/all_new_data_1.xlsx')
data.columns
datatypes = data.dtypes
print(datatypes)
X = data[['vec', 'diffn', 'diffr', 'mixingh', 'mixinge']]
y = data[['phase']]
# load and summarize the dataset
from pandas import read_csv
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
# define the dataset location


# split into input and output elements
#X, y = data[0:4], data[:, -1]
# label encode the target variable
y = LabelEncoder().fit_transform(y)
# summarize distribution
counter = Counter(y)
for k,v in counter.items():
	per = v / len(y) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.show()

# example of oversampling a multi-class classification dataset
from pandas import read_csv
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder

# label encode the target variable
y = LabelEncoder().fit_transform(y)
# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
# summarize distribution
counter = Counter(y)
for k,v in counter.items():
	per = v / len(y) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
kfold = model_selection.KFold(n_splits=10, random_state=None)

classifiers = [['Neural Network :', MLPClassifier(max_iter = 1000)],
               ['LogisticRegression :', LogisticRegression(max_iter = 1000)],
               ['ExtraTreesClassifier :', ExtraTreesClassifier()],
               ['DecisionTree :',DecisionTreeClassifier()],
               ['RandomForest :',RandomForestClassifier()], 
               ['Naive Bayes :', GaussianNB()],
               ['KNeighbours :', KNeighborsClassifier()],
               ['SVM :', SVC()],
               ['AdaBoostClassifier :', AdaBoostClassifier()],
               ['GradientBoostingClassifier: ', GradientBoostingClassifier()],
               ['XGB :', XGBClassifier()],
               ['CatBoost :', CatBoostClassifier(logging_level='Silent')]]

predictions_df = pd.DataFrame()
predictions_df['action'] = y_test

for name,classifier in classifiers:
    classifier = classifier
    classifier.fit(X_train, y_train.ravel())
    predictions = classifier.predict(X_test)
    predictions_df[name.strip(" :")] = predictions
    print(name, accuracy_score(y_test, predictions))
    
    from sklearn.ensemble import VotingClassifier
clf1 = ExtraTreesClassifier()
clf2 = CatBoostClassifier(logging_level='Silent')
clf3 = RandomForestClassifier()
clf4 = DecisionTreeClassifier()
clf5 = KNeighborsClassifier()
clf6 = XGBClassifier()
eclf1 = VotingClassifier(estimators=[('ExTrees', clf1), ('CatBoost', clf2), ('RF', clf3), ('DT', clf4), ('knn', clf5), ('xgb', clf6)], voting='soft')
eclf1.fit(X_train, y_train)
predictions = eclf1.predict(X_test)
print(classification_report(y_test, predictions))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,eclf1.predict(X_test))
ac = accuracy_score(y_test,eclf1.predict(X_test))
print('Accuracy is Logistic: ',ac)
sns.heatmap(cm,annot=True,fmt="d")


import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


import matplotlib.pyplot as plt
 
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()
