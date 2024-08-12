# Confusion Matrix
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, recall_score, average_precision_score, confusion_matrix
from sklearn.inspection import permutation_importance
from added_features_clf import X_df_copy, model_new, y_test2, preds2
from added_features_clf import X2, y2
# conf_matrix = confusion_matrix(y_true=y_test2, y_pred=preds2)
# # Printing Confusion Matrix
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
print(f'Precision: {round(average_precision_score(y_test2, preds2), 3)}')

# Recall
print(f'Recall: {round(recall_score(y_test2, preds2), 3)}')

# F-measure
print(f'F-measure: {round(f1_score(y_test2, preds2), 3)}')

# Plotting feature importances
# importances = {'Feature Name': X_df_copy.columns, 'Feature Importance':model_new[1].feature_importances_}
# importances_df = pd.DataFrame(importances)
# importances_df.sort_values(by='Feature Importance', axis=0, ascending=False, ignore_index=True)
importances = permutation_importance(model_new, X2, y2, n_repeats=10, random_state=0)
print(f"importances{importances}")
