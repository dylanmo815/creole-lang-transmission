# Define Tree Classifier

import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, average_precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score

from baseline import X_df, data_test

clf = RandomForestClassifier(class_weight='balanced_subsample', max_depth=4,
                             max_features="sqrt", min_samples_leaf=2, n_estimators=1000, bootstrap=False, random_state=3)
# Imbalance correction
smote = SMOTE(sampling_strategy='not majority')
model = Pipeline(steps=[('sampling', smote),
                        ('classifier', clf)])
# train test split, target is "class" (creole or not), features, anything not class in tree_features
X = X_df
y = data_test['class']
# encoding targets Creole and WALS to 1 and 0 respectively
y = np.array([{"Creole": 1, "WALS": 0}[y] for y in y.tolist()])

# Building clf classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(f'Decision Tree accuracy is: {sklearn.metrics.accuracy_score(preds, y_test)}')

# Cross validation
scores = cross_val_score(model, X, y, cv=10, scoring='f1_macro')
print("Avg :", np.average(scores))

# precision is simply the ratio of correct positive predictions out of all positive predictions made
# The recall is intuitively the ability of the classifier to find all the positive samples.
# The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta
# score reaches its best value at 1 and worst score at 0.

# Confusion Matrix
conf_matrix = confusion_matrix(y_true=y_test, y_pred=preds)

# Printing Confusion Matrix
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(conf_matrix, cmap=plt.cm.BuPu, alpha=0.3)
# for i in range(conf_matrix.shape[0]):
#     for j in range(conf_matrix.shape[1]):
#         ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
#
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
# plt.show()

# Precision
print(f'Precision: {round(average_precision_score(y_test, preds), 3)}')

# Recall
print(f'Recall: {round(recall_score(y_test, preds), 3)}')

# F-measure
print(f'F-measure: {round(f1_score(y_test, preds), 3)}')

# Feature Importances
# baseline_importances = {'Feature Name': X_df.columns, 'Feature Importance': model[1].feature_importances_}
# baseline_importances_df = pd.DataFrame(baseline_importances)
# baseline_importances_df.sort_values(by='Feature Importance', axis=0, ascending=True, ignore_index=True)
# sorted = baseline_importances_df.sort_values(['Feature Importance'],ascending=True, ignore_index=True)
# print(sorted)
#
# # Exporting
# baseline_importances_df.to_excel('/Users/dylanmoses/Documents/creole_data/baseline_importances_df.xlsx')
new_importances = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0)
print(new_importances)
