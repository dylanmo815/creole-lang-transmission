import pandas as pd

from baseline import data_test

# running same tests on different lexifiers across creoles
# idea is 3 lexifier groups (germanic, romance, other) differ on amount/complexity of morph (other > romance > germanic) so if creoles in each lexifier group show
# similar amounts of morph features, proves further that morphological and tonal features do not pass down to creoles

# so, group all the same lexifiers into their own df
# check out each morphological feature score to see how complex they are
# I guess complexity is just looking at their scores

# making a df with wals code, all morph features, and lexifier
germanic_df = pd.DataFrame
for i in data_test['lexifiers'] == 'Germanic'