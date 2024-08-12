import numpy as np
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, recall_score, average_precision_score, confusion_matrix
from added_features_clf import X_df_copy, data_test_copy

# Grabbing only tonal and morphological features to run through a clf, just for fun ig
from added_features_clf_graph import importances_df

X_df_morph = X_df_copy.iloc[:, -4:]
tone = X_df_copy['Tone']
X_df_morph.insert(0, 'Tone', tone)
locus = X_df_copy['Locus of marking in pos. NP']
X_df_morph.insert(1, 'Locus of marking in pos. NP', locus)

# clf
# Define Tree Classifier
clf_morph = RandomForestClassifier(class_weight='balanced_subsample', max_depth=4,
                                   max_features='auto', min_samples_leaf=2, n_estimators=1000, random_state=3)

# Imbalance correction
smote2 = SMOTE(sampling_strategy='not majority')
model_new = Pipeline(steps=[('sampling', smote2),
                            ('classifier', clf_morph)])

# train test split, target is "class" (creole or not), features, anything not class in tree_features
X_morph = X_df_morph
y_moprh = data_test_copy['class']
# encoding targets Creole and WALS to 1 and 0 respectively
y_moprh = np.array([{"Creole": 1, "WALS": 0}[y_moprh] for y_moprh in y_moprh.tolist()])

# Building clf classifier
X_train_morph, X_test_moprh, y_train_moprh, y_test_moprh = train_test_split(X_morph, y_moprh, test_size=0.4)
model_new.fit(X_train_morph, y_train_moprh)
preds_morph = model_new.predict(X_test_moprh)
print(f'Morph/Tone Decision Tree accuracy is: {sklearn.metrics.accuracy_score(preds_morph, y_test_moprh)}')

scores2 = cross_val_score(model_new, X_morph, y_moprh, cv=10, scoring='f1_macro')
print("Morph/Tone Avg :", np.average(scores2))

# Confusion Matrix
conf_matrix_moprh = confusion_matrix(y_true=y_test_moprh, y_pred=preds_morph)
# Printing Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix_moprh, cmap=plt.cm.BuPu, alpha=0.3)
for i in range(conf_matrix_moprh.shape[0]):
    for j in range(conf_matrix_moprh.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix_moprh[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix Morphological and Tonal Features', fontsize=18)
plt.show()
# Precision
print(f'Precision: {round(average_precision_score(y_test_moprh, preds_morph), 3)}')

# Recall
print(f'Recall: {round(recall_score(y_test_moprh, preds_morph), 3)}')

# F-measure
print(f'F-measure: {round(f1_score(y_test_moprh, preds_morph), 3)}')

# Plotting feature importances
importances_morph = {'Feature Name': X_df_morph.columns, 'Feature Importance': model_new[1].feature_importances_}
importances_df_moprh = pd.DataFrame(importances_morph)
importances_df_moprh.sort_values(by='Feature Importance', axis=0, ascending=False, ignore_index=True)
importances_df.sort_values(by='Feature Importance', axis=0, ascending=False, ignore_index=True)

importances_df_moprh.to_excel('/Users/dylanmoses/Documents/creole_data/morph_df.xlsx')
# importances_df.to_excel('/Users/dylanmoses/Documents/creole_data/importances_df.xlsx')

