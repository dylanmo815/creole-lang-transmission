### Adding new columns to table of morphological features ###
import numpy as np
import pandas as pd

from baseline import X_df

added_features = pd.DataFrame()

# associative_plural = [2, 4, 4, 1, 3, ]
mark_pro_poss = [1, 1, 2, 1, 4, 4, 2, 4, 1, 2, np.nan, 4, 4, 2, 4, 4, 4, 4, 2, np.nan, 4, 4, 1, 4, 4, 4, 4, 2, 4, np.nan, 4, 4, 4, 4, 4, 4, 4, 2, 2, 4, 2, 4, 4, 4, 2, 4, 4, 2, 1, 4, 4, 4, 4, 4, 2, 4, np.nan, 1, np.nan, 2, 4, 4, 1, 4, 4, 4, 4, 2, 1, 4, 4, 4, np.nan, 1, 4, 2, 4, np.nan, 4, 4, 2, 1, 4, 2, 2, 1, np.nan, 4, 4, 4, 1, 4, 4, 2, np.nan, 4, 1, 4, 4, 1, 4, 4, 4, 4, 1, 1, 4, 1, 1, 4, 2, 4, 2, 1, 4, 4, 4, 4, 1, 4, 4, 1, np.nan, 4, 4, np.nan, 4, 3, 4, np.nan, 4, 4, 4, 1, 4, 4, 2, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 2, np.nan, np.nan, np.nan, 4, 4, 4, 4, 3, 4, 2, 4, 2, 1, 1, 1, 1, 4, np.nan, 2, 4, 4]

added_features['marking of pronom. poss'] = mark_pro_poss

mark_poss_np = [3, 3, 2, 3, 1, 1, 3, 2, 3, np.nan, 1, 2, 1, 2, 1, 3, 1, 2, 1, 1, 3, 1, 1, 3, 1, 1, 1, 3, 2, np.nan, np.nan, 3, 2, 1, 1, 1, 2, 1, 1, 1, 1, np.nan, 1, 2, 1, 1, 1, 1, 3, 1, 1, 2, 2, 2, 1, 2, 1, 3, 1, 3, np.nan, np.nan, 1, 1, 2, 1, np.nan, np.nan, 3, 1, 1, 2, 1, 3, 1, 1, np.nan, 1, np.nan, 1, 1, 3, 1, 1, 1, 3, 1, 1, 3, 1, 3, 1, np.nan, 3, 3, np.nan, 3, 1, 1, 3, 2, 1, 1, 3, 3, 2, 2, 1, 1, 1, 1, 2, 1, 1, np.nan, 2, 2, 1, np.nan, 2, 2, 3, 1, 1, 1, np.nan, 1, 3, 2, 2, 2, 3, 1, 1, 2, 1, 1, 2, 2, 2, 1, 3, 1, 1, 2, 1, 3, 1, np.nan, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 3, 3, 3, 1, 3, 1, 1, 2, 3, 1]
# apics 3 = wals 1
# apics 1 = wals 2,3,5
# apics 2 = wals 4
added_features['marking of poss NP'] = mark_poss_np

coding_nom_pl = [2, 2, 6, 2, 8, 5, 6, 7, 2, 2, 2, 2, 2, 8, 6, 5, 6, 6, 6, 7, 2, 8, 2, 6, 2, 2, 7, 7, 2, np.nan, 2, 6, 7, 7, 7, 2, 8, 2, 8, 9, 2, 8, 2, 2, 8, 2, 2, 2, 7, 8, 2, 2, 8, 7, 2, 2, 2, 7, np.nan, 2, 2, 6, 7, 9, 5, 2, 6, 2, 7, 2, 2, 8, 9, 2, 2, 6, 7, 2, 7, 1, 6, 2, 2, 8, 2, 8, 2, 9, 7, 7, 2, 1, 8, 1, 2, 2, 2, 2, 1, 7, 1, 9, 7, 7, 3, 7, 7, 9, np.nan, 2, 2, 7, 2, 2, 7, 7, 7, 2, 2, 7, 7, 2, 1, 9, 7, 7, 7, 9, 7, 9, 7, 2, 2, 8, 7, 2, 1, 2, 7, 7, 2, 2, 7, 2, 7, 2, 2, 1, 8, 7, 7, np.nan, 2, 7, 7, 7, 9, 2, 7, 7, 2, 9, 2, 2, 2, 2, 2, 7, 7]

added_features['coding_nom_pl'] = coding_nom_pl

reduplication = [1, np.nan, 1, 2, 1, 2, 1, 1, 3, 2, 1, 1, 3, 2, 2, 1, 1, 1, 1, 2, 3, 1, np.nan, 1, 3, 1, 2, 1, 1, np.nan, 1, 1, 2, 2, 1, 3, 1, 3, 1, 1, 3, 1, 1, 1, 1, np.nan, 3, 3, np.nan, 2, 1, 2, 3, 1, 1, 3, 1, 3, 1, 1, 1, 1, 3, np.nan, 2, np.nan, 3, np.nan, 1, 2, 1, np.nan, 1, np.nan, np.nan, 1, 1, 1, 1, 2, 2, 3, 1, np.nan, 1, 1, 1, 1, 2, 2, 1, 3, np.nan, 3, 3, np.nan, 1, 3, 2, 1, 2, 1, 1, np.nan, 1, 1, 1, 1, 1, 1, np.nan, 2, 1, np.nan, 1, 3, 2, 1, 1, 1, 1, 2, 1, 1, 3, 1, 1, 1, 2, 3, 2, 1, 1, 2, np.nan, 1, 1, 1, 1, 2, np.nan, 3, np.nan, 3, 1, 1, np.nan, 1, np.nan, 1, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, np.nan, np.nan, 3, 1, 1, 3, 1, 2]

added_features['reduplication'] = reduplication

added_features.head()

# Merging new feature dataframe with old dataframe to rerun analysis
X_df_copy = X_df.copy()

X_df_copy['marking of nom poss'] = mark_pro_poss

X_df_copy['marking of poss NP'] = mark_poss_np

X_df_copy['coding nom pl'] = coding_nom_pl

X_df_copy['reduplication'] = reduplication

print(X_df_copy.shape)