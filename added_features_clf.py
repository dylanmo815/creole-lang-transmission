# Final Random Forest Analysis with additional morphological features
# Missing data and imputations for Decision Tree features
import numpy as np
import pandas as pd
import sklearn
import eli5
# from sns import sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, average_precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from eli5.sklearn import PermutationImportance
from added_features import mark_pro_poss, mark_poss_np, coding_nom_pl, reduplication
from baseline import data_test

# Adding features to whole df (copy)
data_test_copy = data_test.copy()
data_test_copy['marking of nom poss'] = mark_pro_poss

data_test_copy['marking of poss NP'] = mark_poss_np

data_test_copy['coding nom pl'] = coding_nom_pl

data_test_copy['reduplication'] = reduplication

# features to be used in the random forest analysis
all_features_new = data_test_copy.columns.to_list()
unwanted = ['complexity_op',
            'lexifier',
            'substrate',
            'region',
            'sub_class',
            'class',
            'all_features',
            'Atlantic',
            'Biclans..Biclan_name',
            'Bifamilies..Bifamily_name',
            'Lexifier_30WALS',
            'Substrate_30WALS',
            'lexifier test',
            'substrate test']
# list of new feature set
all_features_new = [ele for ele in all_features_new if ele not in unwanted]
tree_features_new = all_features_new[2:]

# Final Data Table for clf
tree_features_df_new = data_test_copy.iloc[:, 2:]

tree_features_df_new.drop(['complexity_op',
                           'lexifier',
                           'substrate',
                           'region',
                           'sub_class',
                           'class',
                           'all_features',
                           'Atlantic',
                           'Biclans..Biclan_name',
                           'Bifamilies..Bifamily_name',
                           'Lexifier_30WALS',
                           'Substrate_30WALS',
                           'lexifier test',
                           'substrate test'], inplace=True, axis=1)

# plt.figure(figsize = (10,6))
# plt.title('Both Creoles & Non-Creoles')
# sns.heatmap(tree_features_df.isna(), cbar=False, yticklabels=False)

# Number of missing data across both creoles and non-creoles
# perc_missing = added_features.isnull().sum() * 100 / len(added_features)
# print(perc_missing)

# average missing data across both creoles and non-creoles
# missing_listed = perc_missing.tolist()
# print(f'All data percent missing {sum(missing_listed)/len(missing_listed)}') # 0.


# Creoles missing data
# creoles_missing = data_test_copy[(data_test_copy['class'] == 'Creole')]
# creoles_missing = creoles_missing.iloc[:, 2:]
# plt.figure(figsize=(10, 6))
# plt.title('Creoles')
# sns.heatmap(creoles_missing.isna(), cbar=False, yticklabels=False)
#
# # Number of missing data across both creoles and non-creoles
# creoles_percent_missing = creoles_missing.isnull().sum() * 100 / len(creoles_missing)
# print(creoles_percent_missing)
#
# # average missing data across both creoles and non-creoles
# creoles_missing_listed = creoles_percent_missing.tolist()
# print(f'Creole data percent missing {sum(creoles_missing_listed) / len(creoles_missing_listed)}')  # 0.0353
#
# # Non-creoles missing data
# else_missing = data_test_copy[(data_test_copy['class'] != 'Creole')]
# else_missing = else_missing.iloc[:, 2:63]
# plt.figure(figsize=(10, 6))
# plt.title('Non-Creoles')
# sns.heatmap(else_missing.isna(), cbar=False, yticklabels=False)
#
# # Number of missing data across both creoles and non-creoles
# else_percent_missing = else_missing.isnull().sum() * 100 / len(else_missing)
# # print(else_percent_missing)
#
# # average missing data across both creoles and non-creoles
# else_missing_listed = else_percent_missing.tolist()
# print(f'Non-creole data percent missing {sum(else_missing_listed) / len(else_missing_listed)}')  # 0.1971
#
# Imputing missing values
imp = SimpleImputer(missing_values=np.NaN)

# Fit transform testing dataframe
X_df_copy = pd.DataFrame(imp.fit_transform(tree_features_df_new))
X_df_copy.columns = tree_features_df_new.columns
X_df_copy.index = tree_features_df_new.index

# Testing col with most missing data for imputation
X_df_copy['Para-linguistic clicks'].isna().sum()

# FINAL RANDOM FOREST ANALYSIS

# Define Tree Classifier
clf_new = RandomForestClassifier(class_weight='balanced', max_depth=4,
                                 max_features='sqrt', min_samples_leaf=2, n_estimators=1000, bootstrap=False,
                                 random_state=3)
# changed class weight from balanced_subsample, kept bootstrap at false
# Imbalance correction
smote2 = SMOTE(sampling_strategy='not majority')
model_new = Pipeline(steps=[('sampling', smote2),
                            ('classifier', clf_new)])

# train test split, target is "class" (creole or not), features, anything not class in tree_features
X2 = X_df_copy
y2 = data_test_copy['class']
# encoding targets Creole and WALS to 1 and 0 respectively
y2 = np.array([{"Creole": 1, "WALS": 0}[y2] for y2 in y2.tolist()])

# Building clf classifier
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.4)
model_new.fit(X_train2, y_train2)
preds2 = model_new.predict(X_test2)
print(f'Decision Tree accuracy is: {sklearn.metrics.accuracy_score(preds2, y_test2)}')

scores2 = cross_val_score(model_new, X2, y2, cv=10, scoring='f1_macro')
print("Avg :", np.average(scores2))

# feature importances
# Feature Importances
# added_importances = {'Feature Name': X_df_copy.columns, 'Feature Importance': model_new[1].feature_importances_}
# baseline_importances_df = pd.DataFrame(added_importances)
# baseline_importances_df.sort_values(by='Feature Importance', axis=0, ascending=True, ignore_index=True)
# sorted = baseline_importances_df.sort_values(['Feature Importance'],ascending=True, ignore_index=True)
# print(sorted)
