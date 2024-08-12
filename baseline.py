# keeping class creoles
import gower
import numpy as np
import pandas as pd
import sch as sch
import sns as sns
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer

from main import data_test_prepared

distances = data_test_prepared.iloc[:, 1:]
distances = distances[(distances['class'] == 'Creole')]
distances.fillna(0, inplace=True)

# gowers distance matrix
# Gowers is a test of *dissimilarity*, so values over .4 would show dissimilar feature sets while any values under
# this heuristic would be similar creoles
gowers = gower.gower_matrix(distances)
print(f"Gower's mean is {gowers.mean()}")
print(gowers)

# hierarchical clustering graph to show results
# length of vertical and horizontal lines show the distance between similarity
# for langauge labels on axis - labels = list(distances.Language)
# dendrogram = sch.dendrogram(sch.linkage(gowers, method='ward'), labels=list(distances.wals_code))
# plt.title("Dendrogram Clustering Measure")
# plt.xlabel('Creoles', fontsize=20)
# plt.xticks(fontsize=20)
# plt.ylabel("Manhattan Distance")  # or eucledian
# plt.yticks([0, .4, .8, 1, 1.5, 2.0, 2.5, 3])
# plt.yticks(fontsize=12)
# plt.rcParams['figure.figsize'] = [100, 75]
# plt.show()
# plt.savefig('/Users/dylanmoses/Documents/creole_data/distance_tree')


# here are the related creoles
# for related groups, keep the one with the most amount of data points,
# ie least about of NaNs. If tied, randomly select


# Kikongo and Lingala
data_test_prepared[(data_test_prepared['wals_code']=='Kikongo-Kituba')].isna().sum().sum()
data_test_prepared[(data_test_prepared['wals_code']=='Lingala')].isna().sum().sum()
# both 2, randomly select Kikongo-Kituba




# Cape Verdean Creole of Brava, Santiago, Sao Vincente
data_test_prepared[(data_test_prepared['wals_code']=='Cape Verdean Creole of Brava')].isna().sum().sum()
data_test_prepared[(data_test_prepared['wals_code']=='Cape Verdean Creole of Sao Vincente')].isna().sum().sum()
data_test_prepared[(data_test_prepared['wals_code']=='Cape Verdean Creole of Santiago')].isna().sum().sum()
# Brava and Santiago = 3, Sao Vincente = 0, keep Sao Vincente

# Seychelles, Reunion Creole, Mauritian Creole
data_test_prepared[(data_test_prepared['wals_code']=='Mauritian Creole')].isna().sum().sum()
data_test_prepared[(data_test_prepared['wals_code']=='Seychelles')].isna().sum().sum()
data_test_prepared[(data_test_prepared['wals_code']=='Reunion Creole')].isna().sum().sum()
# Mauritian and Seychelles both 0, Reunion 1, randomly select Seychelles

# Guadeloupe Creole and Martinician Creole
data_test_prepared[(data_test_prepared['wals_code']=='Guadeloupean Creole')].isna().sum().sum()
data_test_prepared[(data_test_prepared['wals_code']=='Martinican Creole')].isna().sum().sum()
# Both 0, randomly select Martinician

# Santome and Angolar
data_test_prepared[(data_test_prepared['wals_code']=='Santome')].isna().sum().sum()
data_test_prepared[(data_test_prepared['wals_code']=='Angolar')].isna().sum().sum()
# Santome 2, Angolar 0, keep Angolar

# Cavite Chabacano, Zamboagan Chabacano, Ternate Chabacano
data_test_prepared[(data_test_prepared['wals_code']=='Cavite Chabacano')].isna().sum().sum()
data_test_prepared[(data_test_prepared['wals_code']=='Zambaonga Chabacano')].isna().sum().sum()
data_test_prepared[(data_test_prepared['wals_code']=='Ternate Chabacano')].isna().sum().sum()
# keep Ternate

# San Andreas Creole English and Jamaican
data_test_prepared[(data_test_prepared['wals_code']=='Jamaican')].isna().sum().sum()
data_test_prepared[(data_test_prepared['wals_code']=='San Andres Creole English')].isna().sum().sum()
# both 0, randomly select Jamaican

# Creolese and Vincentian Creole
data_test_prepared[(data_test_prepared['wals_code']=='Creolese')].isna().sum().sum()
data_test_prepared[(data_test_prepared['wals_code']=='Vincentian Creole')].isna().sum().sum()
# Keep Vincentian Creole

# Filter data with similar creoles selected above
# data_test = data_test_prepared.drop([110, 27, 28, 119, 153, 59, 158, 31, 156,40])
data_test = data_test_prepared.drop([117, 138, 160, 175, 20, 184, 80, 27, 28])

data_test['class'].value_counts()['Creole']
data_test['class'].value_counts()['WALS']

# Creole      WALS
#   48        120

# Save counts of creoles and wals
num_creoles = data_test['class'].value_counts()['Creole']
num_wals = data_test['class'].value_counts()['WALS']

# Save language names used for analysis
language_names = data_test['wals_code'].to_list()
# 168 languages in total


# Counts of each lexifier
# data_test.iloc adds lexifier to new lexifier_test col in data_test
data_test['lexifier test'] = ''
lexifiers = data_test['lexifier'].to_list()
g_count = 0
r_count = 0
other_count = 0
for i in lexifiers:
    if i in ['English', 'Dutch']:
        g_count += 1
        data_test.loc[(data_test['lexifier'] == i), 'lexifier test'] = 'Germanic'
    elif i in ['Portuguese', 'Spanish', 'French']:
        r_count += 1
        data_test.loc[(data_test['lexifier'] == i), 'lexifier test'] = 'Romance'
    elif i in ['Arabic', 'Bantu', 'Malay']:
        other_count += 1
        data_test.loc[(data_test['lexifier'] == i), 'lexifier test'] = 'Other'
    else:
        data_test.loc[(data_test['substrate'] == i), 'lexifier test'] = 'NaN'
print(f'Germanic Count is {g_count}')  # 23
print(f'Romance Count is {r_count}')  # 21
print(f'Other Count is {other_count}')  # 4

data_test['substrate test'] = ''
substrate = data_test['substrate'].to_list()
au_count = 0
macro_sudan_count = 0
others_count = 0
for i in substrate:
    if i in ['Oceanic, Melanesian, Austronesian', 'Western Malayo-Polynesian', 'Oceanic', 'Central Malayo-Polynesian']:
        au_count += 1
        data_test.loc[(data_test['substrate'] == i), 'substrate test'] = 'Austronesian'
    elif i in ['Atlantic, Mande', 'Ijoid', 'Benue/Kwa', 'Benue/Kwa, Fula. Mende, Vai',
               'Akan, also Cross River (Ibibio), Bantoid (Kituba, Swahili', 'Edo, Yoruba, Kikongo', 'Bantu']:
        macro_sudan_count += 1
        data_test.loc[(data_test['substrate'] == i), 'substrate test'] = 'Macro Sudanese'
    elif i in ['Arabic', 'Bantu', 'Malay', 'Indic', 'Malay, Sinitic', 'Malagasy/Eastern Bantu',
               'Hawaiian, Chinese, Portuguese', 'Nilotic']:
        others_count += 1
        data_test.loc[(data_test['substrate'] == i), 'substrate test'] = 'Other'

# Filling in blanks with NaN
data_test = data_test.mask(data_test == '')

print(f'Austronesian Count is {au_count}')  # 5
print(f'Macro Sudanese Count is {macro_sudan_count}')  # 29
print(f'Other Count is {others_count}')  # 7

languages = data_test['wals_code'].to_list()

# features to be used in the random forest analysis #
all_features = data_test_prepared.columns.to_list()
tree_features = all_features[2:48]


# Missing data and imputations for Decision Tree features

tree_features_df = data_test.iloc[:, 2:48]
plt.figure(figsize = (10,6))
plt.title('Both Creoles & Non-Creoles')
# sns.heatmap(tree_features_df.isna(), cbar=False, yticklabels=False)

# Number of missing data across both creoles and non-creoles
perc_missing = tree_features_df.isnull().sum() * 100 / len(tree_features_df)
print(perc_missing)

# average missing data across both creoles and non-creoles
missing_listed = perc_missing.tolist()
print(f'All data percent missing {sum(missing_listed)/len(missing_listed)}') # 0.1509


# Creoles missing data
creoles_missing = data_test[(data_test['class']=='Creole')]
creoles_missing = creoles_missing.iloc[:, 2:48]
# plt.figure(figsize = (10,6))
# plt.title('Creoles')
# sns.heatmap(creoles_missing.isna(), cbar=False, yticklabels=False)

# Number of missing data across both creoles and non-creoles
creoles_percent_missing = creoles_missing.isnull().sum() * 100 / len(creoles_missing)
print(creoles_percent_missing)

# average missing data across both creoles and non-creoles
creoles_missing_listed = creoles_percent_missing.tolist()
print(f'Creole data percent missing {sum(creoles_missing_listed)/len(creoles_missing_listed)}') # 0.0353



# Non-creoles missing data
else_missing = data_test[(data_test['class'] !='Creole')]
else_missing = else_missing.iloc[:, 2:48]
plt.figure(figsize = (10,6))
plt.title('Non-Creoles')
# sns.heatmap(else_missing.isna(), cbar=False, yticklabels=False)

# Number of missing data across both creoles and non-creoles
else_percent_missing = else_missing.isnull().sum() * 100 / len(else_missing)
# print(else_percent_missing)

# average missing data across both creoles and non-creoles
else_missing_listed = else_percent_missing.tolist()
print(f'Non-creole data percent missing {sum(else_missing_listed)/len(else_missing_listed)}') # 0.1971

# Imputing missing values

imp=SimpleImputer(missing_values=np.NaN)

# Fit transform dataframe
X_df = pd.DataFrame(imp.fit_transform(tree_features_df))
X_df.columns = tree_features_df.columns
X_df.index = tree_features_df.index

# Testing col with most missing data for imputation
X_df['Para-linguistic clicks'].isna().sum()
print(data_test.head())

# Exporting to Excel
file_name = "DataTest.xlsx"
data_test.to_excel(file_name)
print('Exported to Excel')

