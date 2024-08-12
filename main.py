import pandas as pd
import numpy as np
from numpy import mean
import csv
import matplotlib.pyplot as plt
import sys
import gower
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.impute import SimpleImputer
import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from subprocess import call
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, recall_score, f1_score, confusion_matrix
from numpy import nan
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# importing data
data_combined = pd.read_csv("/Users/dylanmoses/Documents/creole_data 2/data_combined.csv", encoding='cp1252')

# keeping only langs labelled creole and non-crole in WALS (ie; no pidgin or mixed langs)
pidgins_and_mixed = data_combined[(data_combined['class']=='WALS') & (data_combined['class']=='Creole')]
delete_row = data_combined[(data_combined["class"]!='WALS') & (data_combined["class"]!='Creole')].index
data_test = data_combined.drop(delete_row)

# excluding Sango (of debated creole status)
data_test[data_test['wals_code']=="Sango"].index
data_test_prepared = data_test.drop([157], axis=0)

