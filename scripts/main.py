# %% [markdown]
# # Imports and Data loading/Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# variance inflation factor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
# import traintestsplit
from sklearn.model_selection import train_test_split
# import crossval score
from sklearn.model_selection import cross_val_score
# import metrics
from sklearn.metrics import mean_squared_error, r2_score
# set pd options to display all columns
pd.set_option('display.max_columns', None)
figure_number = 1 # initialize figure number
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
original_train = train.copy() # make a copy of the original data
original_test = test.copy() # keep a copy of the original data
print("Data loaded from train/test cleaned csv files.")

features_before_selection = train.columns
feature_count_before_selection = len(features_before_selection) # will be used later to compare the number of features before and after selection

# I want to immedietely correct the column names to make them easier to work with. snake_case is the standard for python.
train.columns = [str(col).replace(' ','_').lower() for col in train.columns]
original_train.columns = [str(col).replace(' ','_').lower() for col in original_train.columns]

# To make sure the columns are the same in both dataframes, I will use the same code for the test dataframe.
test.columns = [str(col).replace(' ','_').lower() for col in test.columns]
original_test = [str(col).replace(' ','_').lower() for col in original_test.columns]

# %%
# Generate the Baseline Model
# The baseline model is the model that is used to compare the performance of the other models. It is the simplest model that can be used to predict the target variable. In this case, the baseline model is the mean of the target variable.
# The baseline model is used to determine if the other models are better than the baseline model. If the other models are better than the baseline model, then the other models are useful. If the other models are not better than the baseline model, then the other models are not useful.

# Baseline Model
baseline_model = train['saleprice'].mean() # 181469.70160897123
# The baseline model is the mean of the target variable. The baseline model is used to determine if the other models are better than the baseline model. If the other models are better than the baseline model, then the other models are useful. If the other models are not better than the baseline model, then the other models are not useful.


# %% [markdown]
#

# %% [markdown]
# # EDA

# %%
train.shape, test.shape

# %%
# Drop all columns with more than 50% missing values (NaN)
train = train.dropna(thresh=len(train)*0.5, axis=1)

#? Note: should I be dropping this from the test set?
test = test.dropna(thresh=len(test)*0.5, axis=1)

# %%
train.shape, test.shape

# %%
train.head()

# %%
train.info()

# %%
train.describe()

# %%
# Examine the distributions for these variables
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.distplot(train['saleprice'], ax=ax[0])
sns.distplot(np.log(train['saleprice']), ax=ax[1])
ax[0].set_title('SalePrice Distribution')
ax[1].set_title('Log SalePrice Distribution')
plt.suptitle(f'Figure no. {figure_number}. SalePrice Distributions (original vs log)')
plt.show();


# %%
# I also would like to check the relationship between lot_frontage and gr_liv_area
figure_number = 2
figure = plt.figure(figsize=(12,6))
sns.scatterplot(x='lot_frontage', y='gr_liv_area', data=train)
plt.title(f'Figure no. Lot Frontage vs Gr Liv Area')
plt.savefig(f'../images/figure_no_{figure_number}_lot_frontage_vs_gr_liv_area.png')
plt.show();

# %% [markdown]
# Lot frontage has an outlier beyond 300 and just under 200.  We will need to deal with these. The histogram shows that we can remove the outliers by capping the lot frontage at 125; however this would result in a loss of 410 rows versus capping at 150 which would result in a loss of 20 rows.  We will cap the lot frontage at 150. Now we have 1660 rows remaining in the dataset.
#
# ```python
# # I want to remove the outliers from the data
# train_tester = train[train['lot_frontage'] < 300]
# train_tester2 = train[train['gr_liv_area'] < 4000]
# train_tester.value_counts()
# ```
#

# %% [markdown]
# ### Testing this out
#
# # I want to remove the outliers from the data
# ```python
# print(f'removing outliers from the data lot_frontage > 300 and gr_liv_area > 4000')
# print(f'lot_frontage > 300')
# print(f'original size of train: {train.shape}')
# orig_rows = train.shape[0]
# train_tester = train[train['lot_frontage'] < 300]
# train_tester2 = train[train['gr_liv_area'] < 4000]
# train_tester.head()
# print(train_tester.shape)
# print(train_tester2.shape)
# train_tester = train_tester[train_tester['gr_liv_area'] < 4000]
# print('removing outliers gr_liv_area < 4000')
# print(train_tester.shape)
# print(f'rows removed from train: {orig_rows - train_tester.shape[0]}')
#
# ```
# output:
#
# ```bash
# removing outliers from the data lot_frontage > 300 and gr_liv_area > 4000
# lot_frontage > 300
# original size of train: (2051, 77)
# (1719, 77)
# (2049, 77)
# removing outliers gr_liv_area < 4000
# (1718, 77)
# rows removed from train: 333
# ```
#
# So, if I removed these outliers from the data I would lose 333 houses.
#
#
#

# %%
train.shape

# %%
# what if I only dropped lot_frontage > 300?
trainee = train[train['lot_frontage'] < 300]
# trainee.shape
# so dropping the outliers didn't help much
train.shape[0] - trainee.shape[0]

# what percent of len(train) is 332?
332/len(train)


# %% [markdown]
# So, if we windsorized our data we would lose 332 houses. which translates to roughly a 16% loss of our our total data.

# %%
# I want to remove the outliers from the data
print(f'removing outliers from the data lot_frontage > 300 and gr_liv_area > 4000')
print(f'lot_frontage > 300')
print(f'original size of train: {train.shape}')
orig_rows = train.shape[0]
train_tester = train[train['lot_frontage'] < 300]
train_tester2 = train[train['gr_liv_area'] < 4000]
train_tester.head()
print(train_tester.shape)
print(train_tester2.shape)
train_tester = train_tester[train_tester['gr_liv_area'] < 4000]
print('removing outliers gr_liv_area < 4000')
print(train_tester.shape)
print(f'rows removed from train: {orig_rows - train_tester.shape[0]}')


# %%
# what does the distribution of saleprice look like if these outliers are removed?
figure_number = 3
figure = plt.figure(figsize=(12,6))
sns.distplot(train_tester['saleprice'])
plt.title('SalePrice Distribution without outliers in lot_frontage and gr_liv_area')
plt.savefig(f'../images/figure_no_{figure_number}_saleprice_minuslotfrontage_grlivarea_outliers_distribution.png')
plt.show();


# %%
train['gr_liv_area'].describe()

# %%
# I want to remove the outliers from the data
train = train[train['lot_frontage'] < 300] # 332 rows removed
train = train[train['gr_liv_area'] < 4000] # 1458 rows removed



# %% [markdown]
# I am interested in the impact that basements have on the saleprice, and to further examine this I want to engineer several features that relate to a house's basement.
#
# We are given the feature `bsmt_exposure` in our data which is a measure of how exposed the basement is. It is a categorical feature with the following values:
# 1. GD - Good Exposure
# 2. AV - Average Exposure (split levels or foyers typically score average or above)
# 3. MN - Mimimum Exposure
# 4. No - No Exposure
#
# I am transforming this feature into a numerical feature by assigning the following values:
# 1. GD - 3 - This is a fully exposed basement.
# 2. AV - 2 - This is a basement that is partially exposed.
# 3. MN - 1 - This is a basement that is minimally exposed and is the least visible besides the below-ground basements in the dataset.
# 4. No - 0 - This is a basement that is not exposed at all.
#
# # Feature Engineering
#
# `basement_presence` - This feature is an impact measure. It takes from the numerical values I assigned to the `bsmt_exposure` feature and multiplies it by the `total_bsmt_sf` feature. This gives us a measure of the impact that the basement has on the saleprice of the house.
#
# ```python
# # I want to create a feature that is the impact of the basement on the saleprice
# train['basement_presence'] = train['bsmt_exposure'] * train['total_bsmt_sf']
# train['basement_presence'].value_counts()
# ```
#

# %%
# Map the ordinal variables to integers in basement exposure
train['bsmt_exposure'] = train['bsmt_exposure'].map({'No': 0, 'Mn': 0.5, 'Av': 1, 'Gd': 2})
test['bsmt_exposure'] = test['bsmt_exposure'].map({'No': 0, 'Mn': 0.5, 'Av': 1, 'Gd': 2}) #& to the test set as well, so columns match.

# %%
# I want to create a feature that is the impact of the basement on the saleprice
# I will do this by multiplying the basement exposure by the basement square footage
try:
    train['bsmt_impact'] = train['bsmt_exposure'] * train['total_bsmt_sf']
    test['bsmt_impact'] = test['bsmt_exposure'] * test['total_bsmt_sf']
except Exception:
    print("Please Run the notebook from the top to the bottom")

# %% [markdown]
#

# %% [markdown]
#

# %% [markdown]
#

# %% [markdown]
# The basement is a tricky part of the house to fully capture, as the data is elusive. Our feature `bsmt_qual` is a measure of the height of the basement ceiling but the values are not numerical. We will need to transform this feature into a numerical feature. We will do this by assigning the following values:
# 1. Ex - Excellent (100+ inches) - 5
# 2. Gd - Good (90-99 inches) - 4
# 3. TA - Typical (80-89 inches) - 3
# 4. Fa - Fair (70-79 inches) - 2
# 5. Po - Poor (<70 inches) - 1
# 6. NA - No Basement - 0
# source: https://www.kaggle.com/c/dsi-us-6-project-2-regression-challenge/data
#
# ```python
# # I want to transform the basement quality feature into a numerical feature
# train['bsmt_qual'] = train['bsmt_qual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
# train['bsmt_qual'].value_counts()
# ```
#
# `basement_height` - This feature is an impact measure. It takes from the numerical values I assigned to the `bsmt_qual` feature and multiplies it by the `total_bsmt_sf` feature. This gives us a measure of the impact that the basement has on the saleprice of the house.
#
# ```python
# # I want to create a feature that is the impact of the basement on the saleprice
# train['basement_volume'] = train['bsmt_qual'] * train['total_bsmt_sf']
# train['basement_volume'].value_counts()
# ```
#
# But How much of the basement is visible to a neighbor? This is a tricky question to answer, but we can get a rough estimate by looking at the `bsmt_exposure` feature and combining it with the new feature `basement_volume` that we created above.
#
# The visible surface area of the basement can be derived by first knowing the volume of the basement and then multiplying that by the ratio of the visible surface area to the total surface area. We can get the ratio of the visible surface area to the total surface area by looking at the `bsmt_exposure` feature. We will do this by assigning the following values:
# * Height of basement ceiling - from `bsmt_qual` feature
# * Visible surface area of basement (in square feet)
#
# visible_surface_area = height_of_basement_ceiling * total_surface_area_of_basement * 0.25 # just a rough estimate
#
# ```python
# # I want to create a feature that is the impact of the basement on the saleprice
# train['basement_visible_surface_area'] = train['basement_volume'] * 0.25 # just a rough estimate
# train['basement_visible_surface_area'].value_counts()
# ```
#

# %%
# I want to transform the basement quality feature into a numerical feature
train['bsmt_qual'] = train['bsmt_qual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
test['bsmt_qual'] = test['bsmt_qual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}) #& test set too

# %%
# I want to create a feature that is the impact of the basement on the saleprice
train['basement_volume'] = train['bsmt_qual'] * train['total_bsmt_sf']
test['basement_volume'] = test['bsmt_qual'] * test['total_bsmt_sf'] #& I need to do this for the test set as well

# %%
# I want to create a feature that is the impact of the basement on the saleprice
train['basement_visible_surface_area'] = train['basement_volume'] * 0.25 # just a rough estimate
test['basement_visible_surface_area'] = test['basement_volume'] * 0.25 #& I need to do this for the test set as well

# %%
train['structure_square_footage'] = train['total_bsmt_sf'] + train['1st_flr_sf'] + train['2nd_flr_sf']
test['structure_square_footage'] = test['total_bsmt_sf'] + test['1st_flr_sf'] + test['2nd_flr_sf'] #& I need to do this for the test set as well
train['structure_square_footage'].describe()

# %%
train.columns

# %%
test.columns

# %% [markdown]
# Something is wrong with a house with 334 square feet. I am going to check the price on the house and see if it is a typo.
#
# ```python
# train[train['structure_square_footage'] == 334]
# ```
# Yes, this house sold for 39300, and is not a typo. I am going to remove this house from the data.
#
# ```python
# train = train[train['structure_square_footage'] != 334]
# ```
#
#

# %%
train = train[train['structure_square_footage'] != 334] # 1 row removed

# %%
# checking the distribution of structure_square_footage
figure_number = 4
figure = plt.figure(figsize=(12,6))
sns.distplot(train['structure_square_footage'])
plt.title(f'Figure no. {figure_number} Structure Square Footage Distribution')
plt.savefig(f'../images/figure_no_{figure_number}_structure_square_footage_distribution.png')
plt.show();

# %% [markdown]
# We also want to engineer the total square footage in the structure. This is the sum of the basement square footage and the first floor and second floor square footage. We will call this feature `structure_square_footage`.
#
# ```python
# # Creating a feature that is the total square footage of the structure
# train['structure_square_footage'] = train['total_bsmt_sf'] + train['1st_flr_sf'] + train['2nd_flr_sf']
# train['structure_square_footage'].describe()
# ```
# The `gr_liv_area` feature is the square footage of the house above ground. It would be interesting to see how many of the houses have squarefootage measures (`structure_square_footage`) that are greater than the square footage of the house above ground (`gr_liv_area`). This is an area for future research.
#
#

# %% [markdown]
# # List of Engineered Features
# 1. `basement_presence` - This feature is an impact measure. It takes from the numerical values I assigned to the `bsmt_exposure` feature and multiplies it by the `total_bsmt_sf` feature. This gives us a measure of the impact that the basement has on the saleprice of the house.
# 2. `basement_volume` - This feature is an impact measure. It finds total volume of the basement by multiplying the `bsmt_qual` feature by the `total_bsmt_sf` feature. This gives us a measure of the impact that the basement has on the saleprice of the house.
# 3. `basement_visible_surface_area` - The visible portion of the basement by rough estimate.
# 4. `structure_square_footage` - The total square footage of the structure. This is the sum of the basement square footage and the first floor and second floor square footage.

# %%
# all the features that have to do with basements:
basement_df = train[['bsmt_exposure', 'total_bsmt_sf', 'bsmt_impact','bsmt_qual','bsmt_cond','bsmtfin_type_1','bsmtfin_type_2']]
basement_df.head()

# %% [markdown]
# To eliminate any potential multicollinearity, I am going to drop the following features that were used to generate the new ones above.
# * `bsmt_exposure`
# * `bsmtfin_sf_1`
# * `bsmtfin_sf_2`
# * `1st_flr_sf`
# * `2nd_flr_sf`
# * `bsmt_qual`
# * `total_bsmt_sf`
#

# %%
used_cols = ['bsmt_exposure', 'bsmt_qual', 'total_bsmt_sf','bsmtfin_sf_1', 'bsmtfin_sf_2','1st_flr_sf', '2nd_flr_sf']
train.drop(columns = ['bsmt_exposure', 'bsmt_qual', 'total_bsmt_sf','bsmtfin_sf_1', 'bsmtfin_sf_2','1st_flr_sf', '2nd_flr_sf'], inplace=True)
test.drop(columns = ['bsmt_exposure', 'bsmt_qual', 'total_bsmt_sf','bsmtfin_sf_1', 'bsmtfin_sf_2','1st_flr_sf', '2nd_flr_sf'], inplace=True)
print(f'Removed columns from train: {used_cols}')
print(f'Removed columns from test: {used_cols}') #& I need to do this for the test set as well so the columns match at the end.

# %%
train['bsmt_impact'].value_counts()

# %%
train.head()

# %%
# I also would like to check the relationship between lot_frontage and gr_liv_area
figure_number = 5
figure = plt.figure(figsize=(12,6))
sns.scatterplot(x='lot_frontage', y='gr_liv_area', data=train)
plt.title(f'Figure no. {figure_number}. Lot Frontage vs Gr Liv Area')
plt.savefig(f'../images/figure_no_{figure_number}_lot_frontage_vs_gr_liv_area.png')
plt.show()

# %%
# show the distribution of lot_frontage
figure_number = 6
figure = plt.figure(figsize=(12,6))
sns.distplot(train['lot_frontage'])
plt.title(f'Figure no. {figure_number}. Lot Frontage Distribution')
plt.savefig(f'../images/figure_no_{figure_number}_lot_frontage_distribution.png')
plt.show();

# %% [markdown]
# ## Some Notes on the Data

# %% [markdown]
# 1. Pool Area has an average of 2.30 which really doesn't mean very much. It should be noted that the max for pool area is 800 and there are 1889 houses with pools in the Ames dataset.
# 2. Some houses have 0 for a feature which means that the house does not have that feature.  For example, if a house has 0 for the 3SsnPorch feature, it means that the house does not have a 3 season porch. Introducing a feature with 0 as a value will bias the model by adding complexity without adding any predictive power. We will need to be aware of this during analysis.
# 3. The features with potentially biasing 0 values are:
#    1. 'mas_vnr_area'
#    2. 'bsmtfin_sf_1'
#    3. 'bsmtfin_sf_2',
#    4. 'bsmt_unf_sf'
#    5. '2nd_flr_sf',
#    6. 'low_qual_fin_sf'
#    7.  'wood_deck_sf',
#    8.  'open_porch_sf',
#    9.  'enclosed_porch',
#    10. '3ssn_porch',
#    11. 'screen_porch',
#    12. 'pool_area'
# 4. The data is not normalized.  We will need to normalize the data before we can proceed with our analysis using lasso/ridge regression.
# 5. The average standard deviation of the features included in the model is 3711.5. This is a large standard deviation. Again, normalization will be needed before we can proceed.
# 6. Lot Area has the largest stdev at 6877.97, if this was standardized to match the other features, the next lowest being 479.85, it may result in better model performance.
# 7. The average mean of the features values in the data is 9302.92 which is another reason that we have to normalize the data.
#    1. Removing saleprice takes the average mean down to 993.55. This is a much more reasonable number. Now the next highest mean is for lot_area and year_remod/add, at 10021.67 and 1982.31 respectively.  This is a much more reasonable number. But it illustrates that years in the data could potentially need to be changed to categorical variables.
# 8. The features: gr_liv_area, 1st_flr_sf, and total_bsmt_sf have similar mean ranges.
# 9. The features: bsmtfin_sf_1, bsmtfin_sf_2, and bsmt_unf_sf have similar mean values.
# 10. bsmt_exposure is 'NA' when the house has no basement.
# 11. We can use bsmt_exposure to determine if a house has a basement or not, and how visible it is to passing neighbors.

# %%
# to prevent any data leakage, I will drop the pid column from the train set
train.drop(columns=['pid'], inplace=True)

# %% [markdown]
# Lot frontage appears to have outliers past around 150. Removing the outliers would result in a loss of 333 houses.  We will need to deal with these outliers or accept the loss of these houses. #todo

# %%
# Show a histogram of the lot frontage
figure = plt.figure(figsize=(12,6))
figure_number = 7
sns.distplot(train['lot_frontage'])
plt.title(f'Figure no. {figure_number}. Lot Frontage Distribution')
plt.savefig(f'../images/figure_no_{figure_number}_lot_frontage_distribution_hist.png')
plt.show();


# %%
train[train['year_built'] <= 2010].head()


# %% [markdown]
# How many houses are there in the dataset that were built after 2006 and before 2008?
# ```python
# train[(train['year_built'] > 2006) & (train['year_built'] < 2008)].shape
# ```
# 77 houses were built between 2006 and 2008

# %%
thetweeners = train[(train['year_built'] > 2006) & (train['year_built'] < 2008)]
thetweeners.describe()

# %% [markdown]
#
# `mas_vnr_area` is the masonry veneer area in square feet. This means that the house has bricks or stone on the exterior walls. Some houses in the dataset have 0 square feet of mas_vnr_area.
#
# ```python
# # just show these houses
# train.loc[(train['mas_vnr_area'].isnull()) | (train['mas_vnr_area'] == 0)]
# ```
# Doing value counts we can see that...
#
# ```python
# train['mas_vnr_area'].value_counts()
# ```
# There are 1152 houses that have 0 square feet of mas_vnr_area. This means there must be other types of veneer on the exterior walls.
#
# The table below shows the results of `original_train['mas_vnr_type'].value_counts()`.  This shows that there are 13 houses that have a veneer type of `BrkCmn` and 630 houses that has a veneer type of `BrkFace`. There are also 168 with `Stone`. Finally, 1218 have no masonry on their exterior veneer.
#

# %% [markdown]
# We can see how many houses were remodeled by examining the difference between `year_built` and `year_remod/add`.  If the difference is 0, then the house was not remodeled.  If the difference is greater than 0, then the house was remodeled.
#
# ```python
# # how many rows in year_remod/add are greater than year_built? This will tell us if we need to drop year_remod/add.
# train[train['year_remod/add'] > train['year_built']].shape[0] # how many homes have been remodeled? --> 801
# ```
#
# ```bash
# 801
# ```
#
# With only 801 houses having been remodeled, I don't want to add any to the model's complexity by including the `year_remod/add` feature.  I will drop this feature from the model. But it should be noted that this feature could be useful in predicting the saleprice of a house, and may prove valuable in further study.
#
# ```python
# # drop year_remod/add
# train.drop(columns=['year_remod/add'], inplace=True)
# test.drop(columns=['year_remod/add'], inplace=True)
# ```
#
#

# %%
# how many rows in year_remod/add are greater than year_built? This will tell us if we need to drop year_remod/add.
train[train['year_remod/add'] > train['year_built']].shape[0] # how many homes have been remodeled?

# %%
# drop year_remod/add
# train.drop(columns=['year_remod/add'], inplace=True)
# test.drop(columns=['year_remod/add'], inplace=True)

# %% [markdown]
# # Executive Summary
#

# %% [markdown]
# ## Question: Which features would neighbors be able to see?
#
# ### Answer: We decided to keep the following variables:
# * Wood Deck SF, Open Porch SF, Enclosed Porch, 3Ssn Porch, Screen Porch, Pool Area, Pool QC, Fence, and Fireplace Qu.
# * The square footage of the above-ground structure is a visual factor that will be considered. This includes the following features: Gr Liv Area, 1st Flr SF, 2nd Flr SF, Low Qual Fin SF, and Garage Area.
#   * for further research we recommend an analysis of any features that could be observed through social media or other online sources that may impact saleprice CCFs. This could include kitchen characteristics, bathroom characteristics, and other features that may be visible in photos. Such a deep dive is beyond the scope of this study.
# * The materials used for the construction of the outside of the house are visible and will be included. This includes the following features: Exterior 1st, Exterior 2nd, Mas Vnr Area, and Mas Vnr Type.
#   * if possible we will assign higher weights to exterior features that are on the second story of the house. This is because the second story is more visible to neighbors than the first story.
# * The final factors that will be considered are `the year the house was built` and the year the house was remodeled. This includes the following features: Year Built, Year Remod/Add, and Yr Sold.

# %%
# CCF Features - the features listed above.
ccf_features_basic = ['lot_area', 'garage_type', 'garage_yr_blt',     'garage_finish','garage_area', 'garage_qual', 'garage_cond',
       'kitchen_qual', 'wood_deck_sf', 'open_porch_sf', 'enclosed_porch',
       '3ssn_porch', 'screen_porch', 'pool_area', 'fireplace_qu',
       'gr_liv_area', 'low_qual_fin_sf', 'year_remod/add',
       'exterior_1st', 'exterior_2nd', 'mas_vnr_area', 'mas_vnr_type',
       'year_built', 'yr_sold','bsmt_impact',
       'basement_volume', 'basement_visible_surface_area',
       'structure_square_footage'] + ['saleprice']

#^ I removed garage_cars because the garage is often closed. I want to see if the garage area is a better predictor of saleprice.
#* garage_cars
#^ I removed the 1st_flr_sf and 2nd_flr_sf because I created a feature that is the sum of the 1st and 2nd floor square footage. Keeping them in would be redundant.
#* '1st_flr_sf', '2nd_flr_sf'
#^ I removed bsmt_exposure because I created a feature that is the impact of the basement on the saleprice.
#* bsmt_exposure
#^ I removed 'year_remod/add' because it is potentially biasing the model without adding enough to the value of the model.
#* 'year_remod/add'
#& pool_qc, fence - did not go through the correlation matrix, but still may have value.

#ccf_features_basic = ['1st_flr_sf','2nd_flr_sf','lot_area'] + ['saleprice']


# %%
# CCF Features - the features listed above.
train = train[ccf_features_basic] # select only the features listed above
test = test[ccf_features_basic[:-1]] # I want to make sure the test set has the same columns as the train set.

# %%
train.head()

# %% [markdown]
# columns to dummify: exterior_1st, exterior_2nd, mas_vnr_type, garage_type, garage_finish, garage_qual, garage_cond, pool_qc, fence, fireplace_qu

# %%
train.drop(columns=['garage_yr_blt'], inplace=True)

# %%
#exterior_1st, exterior_2nd, mas_vnr_type, garage_type, garage_finish, garage_qual, garage_cond, pool_qc, fence, fireplace_qu
dummified_features_df = pd.get_dummies(train, columns=['garage_type', 'garage_finish', 'garage_qual', 'garage_cond', 'kitchen_qual', 'fireplace_qu', 'exterior_1st', 'exterior_2nd', 'mas_vnr_type'], drop_first=True)

# %%
train.info()

# %% [markdown]
# Not enough data on garage_type,garage_finish,or garage_cond, or garage_qual, as well as fireplace_qu
#

# %%
# drop those:
train = train.drop(columns=['garage_type', 'garage_finish', 'garage_qual', 'garage_cond', 'kitchen_qual', 'fireplace_qu'])

# %%
train.head()

# %%
for col in test.columns:
    if col not in train.columns:
        test.drop(columns=col, inplace=True)
        print(f'Dropped {col} from test set.')
        ccf_features_basic.remove(col) # also remove the column from the list of features

# justification: When testing the model on the test set, I want to make sure the test set has the same columns as the train set.

# %%
assert(train.shape[1] == test.shape[1]+1)

# %% [markdown]
# What kinds of distributions do we have in the data?

# %%
train.head()

# %%
# create a line of subplots, one distplot for each feature
figure_number = 8

# center the plots on the figure
ax, fig = plt.subplots(figsize=(30,15))
# only plot numeric features
for i, col in enumerate(train.select_dtypes('number').columns):
    plt.subplot(5, 5, i+1)
    sns.distplot(train[col])
    plt.title(col);
figure.subplots_adjust(hspace=0.5, wspace=0.5)
plt.tight_layout()
plt.suptitle(f'Figure no. {figure_number}. Distribution of Features', fontsize=20, y=1.08)
plt.savefig(f'../images/figure_no_{figure_number}_distplots.png')
plt.show();


# %% [markdown]
# ## Question: Which features matter for my analysis?
#
# ### Answer:
# I will use the correlation matrix to find out which features are highly correlated with the target variable (saleprice).

# %% [markdown]
# # Analysis

# %%
# datatypes
train.dtypes

# %% [markdown]
# I need to dummify exterior_1st, mas_vnr_type, and exterior_2nd.

# %%
dummies = pd.get_dummies(train, columns = ['exterior_1st','exterior_2nd','mas_vnr_type'], drop_first=True)
dummies_test = pd.get_dummies(test, columns = ['exterior_1st','exterior_2nd','mas_vnr_type'], drop_first=True)
dummies.head() # is train + dummies


# %%
# move saleprice to the end again
saleprice = dummies['saleprice']
dummies.drop(columns=['saleprice'], inplace=True)
dummies['saleprice'] = saleprice
dummies_test.to_csv('../data/test_dummies.csv', index=False)
dummies.to_csv('../data/train_dummies.csv', index=False)
test = pd.read_csv('../data/test_dummies.csv') #& I need to do this for the test set as well
# train = pd.read_csv('../data/train_dummies.csv')
train = dummies
dummies.head()

# %%
# remove 'exterior_1st','exterior_2nd' and 'mas_vnr_type', from ccf_features_basic
ccf_features_basic.remove('exterior_1st')
ccf_features_basic.remove('exterior_2nd')
ccf_features_basic.remove('mas_vnr_type')
# Reason: I have dummified these columns, so I don't need them anymore.


# %%
train.head()

# %%
# create a correlation matrix
# of the ccf_features above, I want to determine which are the most highly correlated with saleprice (target).
# I will use the correlation matrix to determine this.
corr_matrix = train[ccf_features_basic].corr()
corr_matrix['saleprice'].sort_values(ascending=False)

# I will use the correlation matrix to determine which features are highly correlated with each other.
# I will use a heatmap to visualize it with a mask over the diagonal.
figure_number  = 9
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix, mask = mask, annot=True, cmap='coolwarm')
# save the plot
plt.title(f'Figure {figure_number}: CCF Correlation Matrix', fontsize=20)
plt.savefig(f'../images/figure_no_{figure_number}_heatmap.png')
plt.show();

# %% [markdown]
# I need to determine which features are highly correlated with each other. I will use a heatmap to visualize it with a mask over the diagonal. This is to eliminate multicollinearity.
#
# Analysis:
# 1. garage_area is highly correlated with garage_cars. To avoid multicollinearity, I will drop garage_cars from the dataset. They have the same correlation with saleprice. Wood deck and open porch also are equally correlated with saleprice. This could indicate a relationship but they are not highly correlated with each other. I will keep both of these features.
# 2. gr_liv_area is highly correlated with 1st_flr_sf and 2nd_flr_sf. I will keep 1st and 2nd floor variables (which sum to more than the correlation of the gr_liv_area corr) and drop gr_liv_area.
#
#

# %%
train.columns

# %%
# dropping columns that are highly correlated with each other as mentioned above.
train.drop(columns=['garage_area'], inplace=True) # highly correlated with garage_cars
train.drop(columns=['gr_liv_area'], inplace=True) # highly correlated with the first and second floor square footage

# also removing them from the list of features
ccf_features_basic.remove('garage_area')
ccf_features_basic.remove('gr_liv_area')

# Dropping base features that were used to create the new features.
# train.drop(columns=['bsmt_exposure', 'bsmtfin_sf_2', 'bsmt_qual', 'total_bsmt_sf','1st_flr_sf','2nd_flr_sf'], inplace=True)

# ccf_features_basic.remove('bsmt_exposure')
# ccf_features_basic.remove('bsmtfin_sf_2')
# ccf_features_basic.remove('bsmt_qual')
# ccf_features_basic.remove('total_bsmt_sf')
# ccf_features_basic.remove('1st_flr_sf')
# ccf_features_basic.remove('2nd_flr_sf')


# %%
train.isnull().sum().sum()

# %% [markdown]
# There are very few features that have missing values the highest being 22, so I feel comfortable performing a dropna now.

# %%
print(f'train.shape: {train.shape}')
train.dropna(inplace=True)
print(f'train.shape: {train.shape}')

# %%
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(train[ccf_features_basic].values, i) for i in range(train[ccf_features_basic].shape[1])]
vif["features"] = train[ccf_features_basic].columns
vif.head(100)

# %% [markdown]
# Any features that have a vif over five will be dropped.
#

# %%
# only include features with a VIF of less than 5 and not equal to 'inf' or np.Inf
# filter vif table to only include features with a VIF of less than 5, that is not salesprice or one of my engineered features and not equal to 'inf' or np.Inf
# vif = vif[((vif['VIF Factor'] < 5) & (vif['VIF Factor'] != np.inf) & (vif['VIF Factor'] != 'inf') & (vif['features'] != 'saleprice')) & ((vif['features'] == 'basement_volume') | (vif['features'] == 'basement_visible_surface_area') | (vif['features'] == 'structure_square_footage'))]

#vif = vif[(vif['VIF Factor'] < 5) & (vif['VIF Factor'] != np.inf) & (vif['VIF Factor'] != 'inf') & (vif['features'] != 'saleprice')]
vif_to_drop = vif[(vif['VIF Factor'] > 5) & (vif['VIF Factor'] != np.inf) & (vif['VIF Factor'] != 'inf') & (vif['features'] != 'saleprice')]
vif_to_drop.head(15)

# %%
# remove features with a VIF of more than 5 from train and ccf_features_basic
print(f'train.shape before removing np.Inf: {train.shape}')
train.drop(columns=vif_to_drop['features'], inplace=True)
for feature in vif_to_drop['features']:
    ccf_features_basic.remove(feature)
print(f'train.shape before removing np.Inf: {train.shape}')

# %%
train.head()

# %%
# remove all np.Inf from train
print(f'train.shape before removing np.Inf: {train.shape}')
train = train.replace(np.inf, np.nan)
print(f'train.shape: {train.shape}')

# %%
train.dtypes

# %%
figure_number = 10
plt.figure(figsize=(25,30))
# make barplot of all features and their correlation with saleprice except for the target
sns.barplot(x=train.corr()['saleprice'].sort_values(ascending=False)[1:], y=train.corr()['saleprice'].sort_values(ascending=False)[1:].index)

plt.yticks(fontsize=20)
plt.title(f'Figure {figure_number}: Correlation of CCF Features with Saleprice', fontsize=20)
plt.savefig(f'../images/figure_no_{figure_number}_barplot.png')
plt.show();


# %% [markdown]
# Creating a figure like the one above to illustrate the coefficients for our linear regression model.
#

# %%
# create a correlation matrix
# of the ccf_features above, I want to determine which are the most highly correlated with saleprice (target).
# I will use the correlation matrix to determine this.
corr_matrix = train[ccf_features_basic].corr()
corr_matrix['saleprice'].sort_values(ascending=False)

# I will use the correlation matrix to determine which features are highly correlated with each other.
# I will use a heatmap to visualize it with a mask over the diagonal.
figure_number = 11
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix, mask = mask, annot=True, cmap='coolwarm',)
# save the plot
plt.xticks(fontsize=20)
plt.tight_layout()
plt.title(f'Figure {figure_number}: CCF Correlation Matrix', fontsize=20)
plt.savefig(f'../images/figure_no_{figure_number}_heatmap.png')

plt.show();

# %% [markdown]
# Due to the way we made the basement_visible_surface_area feature, we need to drop basement_volume to avoid multicollinearity.
#
# ```python
# # drop basement_volume
# train.drop(columns=['basement_volume'], inplace=True)
# ccf_features_basic.remove('basement_volume') # remove it from features array as well.
# ```
#

# %%
# drop basement_volume
train.drop(columns=['basement_volume'], inplace=True)
ccf_features_basic.remove('basement_volume') # remove it from features array as well.

# %%
# create a correlation matrix
# of the ccf_features above, I want to determine which are the most highly correlated with saleprice (target).
# I will use the correlation matrix to determine this.
corr_matrix = train[ccf_features_basic].corr()
corr_matrix['saleprice'].sort_values(ascending=False)

# I will use the correlation matrix to determine which features are highly correlated with each other.
# I will use a heatmap to visualize it with a mask over the diagonal.
figure_number = 12
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix, mask = mask, annot=True, cmap='coolwarm')
# save the plot
plt.title(f'Figure {figure_number}: CCF Correlation Matrix', fontsize=30, y=0.97)
plt.yticks(fontsize=20)
# add some padding to the top of the plot


plt.savefig(f'../images/figure_no_{figure_number}_heatmap.png')

plt.show();

# %%
# keeping only the features with a correlation of 0.25 or higher with saleprice (in positive or negative direction)
ccf_features = corr_matrix[corr_matrix['saleprice'] > 0.25].index.tolist()
print(ccf_features)

# %% [markdown]
# Let's look at just the `saleprice` column as a heatmap to see the correlation between the features and the target variable.
#
#
# ```python
# # create a heatmap column of all features and their correlation to saleprice
# plt.figure(figsize=(12, 12))
# sns.heatmap(train[ccf_features_basic].corr()[['saleprice']].sort_values(by='saleprice', ascending=False), annot=True, cmap='coolwarm')
# plt.title('Correlation of Features to Saleprice')
# plt.show()
# ```
#
#

# %%
# create a heatmap column of all features and their correlation to saleprice
figure_number = 13

plt.figure(figsize=(12, 12))
sns.heatmap(train[ccf_features_basic].corr()[['saleprice']].sort_values(by='saleprice', ascending=False)[1:], annot=True, cmap='coolwarm')
plt.title(f'Figure {figure_number}: Correlation of CCF Features with Saleprice', fontsize=20)
plt.tight_layout()
plt.savefig(f'../images/figure_no_{figure_number}_singlecolumn_heatmap.png')
plt.show()

# %% [markdown]
#

# %% [markdown]
# # Models

# %% [markdown]
# I am going to use a linear regression model to predict the saleprice of a house in Ames, Iowa, and I will use the R2 score to evaluate the model. Secondarily, I will also check a lasso, and ridge regression model to see if they perform better than the linear regression model.

# %%
model_scores = pd.DataFrame() # to hold all model scores
# row format will be: model_name, r2 score, rmse score, train_score, test_score, train_rmse, test_rmse, crossval_score
row = {} # to hold a single row of model_scores


# %%
train = train[ccf_features]

# %%
# the baseline model
# I will use the mean of the saleprice as the baseline model
b = train['saleprice'].mean()

model_predictions = {} # dictionary to store the predictions of the models
model_scores = {} # dictionary to store the scores of the models


# %% [markdown]
# ## Linear Regression Model
#

# %%
# Building our Model
lr = LinearRegression()
X = train.drop(columns=['saleprice'])
y = train['saleprice'] # target

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42) # split the data into train and validation sets

lr.fit(X_train,y_train) # fit on training data
lr.score(X_train, y_train) # score on training data

# Predictions and Evaluation of the Model
y_val_preds = lr.predict(X_val) # predict on validation data

print(f'training score: {lr.score(X_train, y_train)}') # score on training data
print(f'validataion score: {lr.score(X_val, y_val)}') # score on validation data
print(f'cross_val_score: {cross_val_score(lr, X_train, y_train, cv=5).mean()}') # cross_val_score on training data

lr_score = lr.score(X_val, y_val) # score on validation data
model_scores['lr'] = lr_score # add the score to the model_scores dictionary
model_predictions['lr'] = y_val_preds # add the predictions to the model_predictions dictionary

train_score = lr.score(X_train, y_train) # score on training data
test_score = lr.score(X_val, y_val) # score on validation data
cval_score = cross_val_score(lr, X_train, y_train, cv=5).mean() # score on cross validation data





# %% [markdown]
#

# %%
# Plot the predictions vs the actual values
figure_number = 14
model = 'LinearRegression'
plt.figure(figsize=(10, 10))
plt.scatter(y_val, y_val_preds) # plot the predictions vs the actual values
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, ls="--", c=".3")
# remove grid
plt.grid(False)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.annotate(f'model: {model} \ntrain:{train_score}\ntest:{test_score}\ncval:{cval_score}', xy=(0.05, 0.75), xycoords='axes fraction')
plt.title(f'Figure {figure_number}: {model} Predicted vs Actual Sale Price', fontsize=20)
plt.savefig(f'../images/figure_no_{figure_number}_{model}_actual_vs_predicted.png')
plt.show();

# %%
# plot the residuals
figure_number = 15
model = 'LinearRegression'
figure = plt.figure(figsize=(10, 10))
sns.set_style("whitegrid", {'axes.grid' : False})
sns.residplot(y_val_preds, y_val, lowess=True, color="g")
plt.annotate(f'model: {model} \nR2: {lr_score}', xy=(0.1, 0.9), xycoords='axes fraction', fontsize=15)
plt.title(f'Figure {figure_number}: LR Residuals', fontsize=20)
plt.savefig(f'../images/figure_no_{figure_number}_{model}_residuals.png')
plt.show();


# %% [markdown]
#

# %%
# get the coefficients of the model as a dataframe
lr_coefs = pd.DataFrame(lr.coef_, X.columns, columns=["Coefficients"])


# %%

# plot the coefficients
# coefficients[coefficients["Coefficients"] > 0].sort_values(by="Coefficients").plot(kind="barh")
# spread out the y labels

lr_coefs.head(200).sort_values(by="Coefficients", ascending=False)

# %%
# Inference variables
coefficients = lr_coefs # set to linear regression

largest_coef_value = coefficients['Coefficients'].sort_values(ascending=False)[0]
largest_coef_feature = coefficients['Coefficients'].sort_values(ascending=False).index[0]
smallest_coef_value = coefficients['Coefficients'].sort_values(ascending=False)[-1]
smallest_coef_feature = coefficients['Coefficients'].sort_values(ascending=False).index[-1]

# Inference on the coefficients
print(f'A one unit increase in {largest_coef_feature}, holding all other features constant, will result in a {coefficients.loc[largest_coef_feature, "Coefficients"]} increase in saleprice.')

print(f'A one unit decrease in {smallest_coef_feature}, holding all other features constant, will result in a {coefficients.loc[smallest_coef_feature, "Coefficients"]} increase in saleprice.')

largest_coefs_four_lr = coefficients['Coefficients'].sort_values(ascending=False)[0:4]
smallest_coefs_four_lr = coefficients['Coefficients'].sort_values(ascending=False)[-4:]


# %%
model_scores_df = pd.DataFrame() # to hold all model scores

# %%
# add the model scores to the model_scores dataframe
row['model_name'] = 'LinearRegression'
row['r2_score'] = lr_score
row['rmse_score'] = np.sqrt(mean_squared_error(y_val, y_val_preds))
row['train_score'] = train_score
row['test_score'] = test_score
row['cval_score'] = cval_score
row = pd.DataFrame(row, index=[0])
model_scores_df = model_scores_df.append(row, ignore_index=True)
# row format will be: model_name, r2 score, rmse score, train_score, test_score, train_rmse, test_rmse, crossval_score

# %%
model_scores_df.head()

# %% [markdown]
# # Lasso Regression Model

# %%
# Building our Lasso Model
# import lassocv

#! TODO - Scale before Lasso, .....

ss = StandardScaler()


from sklearn.linear_model import LassoCV
# Init, fit, test Lasso Regressor
alphas = np.logspace(-4, 0, 600) # create a list of alphas to test
lasso = LassoCV(alphas=alphas, cv=5) # create a lasso regression model
X = train.drop(columns=['saleprice'])
y = train['saleprice'] # target

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  random_state=42) # split the data into train and validation sets

X_train_lr = X_train
X_val_lr = X_val
X_train = ss.fit_transform(X_train)
X_val = ss.transform(X_val)

lasso.fit(X_train,y_train) # fit on training data


# Predictions and Evaluation of the Model
# y_preds = lasso.predict(X_train) # predict on training data
y_val_preds = lasso.predict(X_val) # predict on validation data

print(f'validataion score: {lasso.score(X_val, y_val)}') # score on validation data

# print(mean_squared_error(y_val, y_preds)) # mean squared error on validation data
# print(r2_score(y_val, y_preds)) # r2 score on validation data

print(f'training score: {lasso.score(X_train, y_train)}') # score on training data

train_score = lasso.score(X_train, y_train) # score on training data
test_score = lasso.score(X_val, y_val) # score on validation data
cval_score = cross_val_score(lasso, X_train, y_train, cv=5).mean() # score on cross validation data

# add the score to the model_scores dictionary
model_scores['lasso'] = lasso.score(X_val, y_val)
# add the predictions to the model_predictions dictionary
model_predictions['lasso'] = y_val_preds


# %%
# Plot the predictions vs the actual values
figure_number = 15
model_name = 'Lasso'
plt.figure(figsize=(10, 10))
plt.scatter(y_val, y_val_preds) # plot the predictions vs the actual values
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, ls="--", c=".3")
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.annotate(f'model: {model} \ntrain:{train_score}\ntest:{test_score}\ncval:{cval_score}', xy=(0.05, 0.75), xycoords='axes fraction')
plt.title(f'Figure {figure_number}: Lasso Predicted vs Actual Sale Price', fontsize=20)
plt.savefig(f'../images/figure_no_{figure_number}_{model_name}_actual_vs_predicted.png')
lasso_score = lasso.score(X_val, y_val)

plt.show()

# %%
# plot the residuals
figure_number = 16
model_name = 'Lasso'
figure = plt.figure(figsize=(10, 10))
sns.set_style("whitegrid", {'axes.grid' : False})
sns.residplot(y_val_preds, y_val, lowess=True, color="g")
plt.ylabel('Residuals')
plt.xlabel('Predicted Sale Price')
plt.title(f'Figure {figure_number}: Lasso Residuals: {lasso.score(X_val,y_val)}', fontsize=20)
plt.savefig(f'../images/figure_no_{figure_number}__{model_name}_residuals.png')
plt.show();


# %%
# get the coefficients of the model as a dataframe
figure_number = 17
figure = plt.figure(figsize=(30, 30))
model_name = 'Lasso'
coefficients = pd.DataFrame(lasso.coef_, X.columns, columns=["Coefficients"])


# %%
# Inference variables
coefficients = coefficients # set to linear regression

largest_coef_value = coefficients['Coefficients'].sort_values(ascending=False)[0]
largest_coef_feature = coefficients['Coefficients'].sort_values(ascending=False).index[0]
smallest_coef_value = coefficients['Coefficients'].sort_values(ascending=False)[-1]
smallest_coef_feature = coefficients['Coefficients'].sort_values(ascending=False).index[-1]

# Inference on the coefficients
print(f'A one unit increase in {largest_coef_feature}, holding all other features constant, will result in a {coefficients.loc[largest_coef_feature, "Coefficients"]} log-unit increase in saleprice.') # log unit increase, because we are using regularized regression

print(f'A one unit decrease in {smallest_coef_feature}, holding all other features constant, will result in a {coefficients.loc[smallest_coef_feature, "Coefficients"]} log-unit increase in saleprice.')

largest_coefs_four_lasso = coefficients['Coefficients'].sort_values(ascending=False)[0:4]
smallest_coefs_four_lasso = coefficients['Coefficients'].sort_values(ascending=False)[-4:]


# %%

# plot the coefficients
figure_number = 18
coefficients[coefficients["Coefficients"] > 0].sort_values(by="Coefficients").plot(kind="barh")
# spread out the y labels
plt.yticks(rotation=0)
plt.title(f'Figure No. {figure_number}: LR Positive Coefficients')
plt.savefig(f"../images/figure_no_{figure_number}_{model_name}_positivecoefficients.png")
plt.show();



# %%
# add the model scores to the model_scores dataframe
row['model_name'] = 'Lasso'
row['r2_score'] = lasso_score
row['rmse_score'] = np.sqrt(mean_squared_error(y_val, y_val_preds))
row['train_score'] = train_score
row['test_score'] = test_score
row['cval_score'] = cval_score
row = pd.DataFrame(row, index=[0])

# row format will be: model_name, r2 score, rmse score, train_score, test_score, train_rmse, test_rmse, crossval_score

# %%

model_scores_df = model_scores_df.append(row, ignore_index=True)


# %% [markdown]
# # Ridge Regression Model

# %%
# Building our Ridge Model
from sklearn.linear_model import Ridge

ridge = Ridge()
X = train.drop(columns=['saleprice'])
y = train['saleprice'] # target

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42) # split the data into train and validation sets

X_train = ss.fit_transform(X_train)
X_val = ss.transform(X_val)

ridge.fit(X_train,y_train) # fit on training data

# Predictions and Evaluation of the Model
y_preds = ridge.predict(X_train) # predict on training data
y_val_preds = ridge.predict(X_val) # predict on validation data


ridge.score(X_train, y_train) # score on training data
ridge_score = ridge.score(X_val, y_val)
print(f'validataion score: {ridge.score(X_val, y_val)}') # score on validation data
print(f'training score: {ridge.score(X_train, y_train)}') # score on training data
train_score = ridge.score(X_train, y_train) # score on training data
test_score = ridge.score(X_val, y_val) # score on validation data
cval_score = cross_val_score(ridge, X_train, y_train, cv=5).mean() # score on cross validation data

print(f'cross_val_score (train): {cross_val_score(ridge, X_train, y_train, cv=5).mean()}') # cross_val_score on training data
print(f'cross_val_score (validation): {cross_val_score(ridge, X_train, y_train, cv=5).mean()}') # cross_val_score on training data

# add the score to the model_scores dictionary
model_scores['ridge'] = ridge.score(X_val, y_val)
# add the predictions to the model_predictions dictionary
model_predictions['ridge'] = y_val_preds


# %%
row['model_name'] = 'Ridge'
row['r2_score'] = ridge.score(X_val, y_val)
row['rmse_score'] = np.sqrt(mean_squared_error(y_val, y_val_preds))
row['train_score'] = train_score
row['test_score'] = test_score
row['cval_score'] = cval_score
row = pd.DataFrame(row, index=[0])
model_scores_df = model_scores_df.append(row, ignore_index=True)

# %%
# Plot the predictions vs the actual values
figure_number = 19
model = 'Ridge'
plt.figure(figsize=(10, 10))
plt.scatter(y_val, y_val_preds) # plot the predictions vs the actual values
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, ls="--", c=".3")
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.annotate(f'model: {model} \ntrain:{train_score}\ntest:{test_score}\ncval:{cval_score}', xy=(0.05, 0.75), xycoords='axes fraction')
plt.title(f'Figure {figure_number}: {model} Predicted vs Actual Sale Price')
plt.savefig(f'../images/figure_no_{figure_number}_{model}_actual_vs_predicted.png')
plt.show();

# %%
# plot the residuals
figure_number = 20
model = 'Ridge'
figure = plt.figure(figsize=(10, 10))
sns.set_style("whitegrid", {'axes.grid' : False})
sns.residplot(y_val_preds, y_val, lowess=True, color="g")
plt.title(f'Figure {figure_number}: {model} Residuals: { ridge.score(X_val, y_val)}', fontsize=20)
plt.annotate(f'model: {model} \nR2: {ridge_score}', xy=(0.1, 0.9), xycoords='axes fraction', fontsize=15)
plt.ylabel('Residuals')
plt.xlabel('Predicted Sale Price')
plt.savefig(f'../images/figure_no_{figure_number}_{model}_residuals.png')

ridge_score = ridge.score(X_val, y_val)

plt.show();


# %%
# get the coefficients of the model as a dataframe
figure_number = 21
model_name = 'Ridge'
coefficients = pd.DataFrame(ridge.coef_, X.columns, columns=["Coefficients"])
coefficients.head()

# %%
# plot the coefficients
coefficients[coefficients["Coefficients"] > 0].sort_values(by="Coefficients").plot(kind="barh")
# spread out the y labels
plt.yticks(rotation=0)
plt.title(f'Figure No. {figure_number}: {model_name} Positive Coefficients {ridge.score(X_val, y_val)}')

plt.savefig(f"../images/figure_no_{figure_number}_{model_name}_positivecoefficients.png")
plt.show();

# %%
# Inference variables
coefficients = coefficients # set to linear regression

largest_coef_value = coefficients['Coefficients'].sort_values(ascending=False)[0]
largest_coef_feature = coefficients['Coefficients'].sort_values(ascending=False).index[0]
smallest_coef_value = coefficients['Coefficients'].sort_values(ascending=False)[-1]
smallest_coef_feature = coefficients['Coefficients'].sort_values(ascending=False).index[-1]

# Inference on the coefficients
print(f'A one unit increase in {largest_coef_feature}, holding all other features constant, will result in a {coefficients.loc[largest_coef_feature, "Coefficients"]} log-unit increase in saleprice.') # log unit increase, because we are using regularized regression

print(f'A one unit decrease in {smallest_coef_feature}, holding all other features constant, will result in a {coefficients.loc[smallest_coef_feature, "Coefficients"]} log-unit increase in saleprice.')

largest_coefs_four_ridge = coefficients['Coefficients'].sort_values(ascending=False)[0:4]
smallest_coefs_four_ridge = coefficients['Coefficients'].sort_values(ascending=False)[-4:]


# %%
lr.coef_

# %% [markdown]
# # Elastic Net Model

# %%

# Building our ElasticNet Model
from sklearn.linear_model import ElasticNet

elasticnet = ElasticNet()
model = 'ElasticNet'

X = train.drop(columns=['saleprice'])
y = train['saleprice'] # target

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42) # split the data into train and validation sets

X_train = ss.fit_transform(X_train)
X_val = ss.transform(X_val)

elasticnet.fit(X_train,y_train) # fit on training data

# Predictions and Evaluation of the Model

y_preds = elasticnet.predict(X_train) # predict on training data
y_val_preds = elasticnet.predict(X_val) # predict on validation data

elasticnet.score(X_train, y_train) # score on training data

print(f'validataion score: {elasticnet.score(X_val, y_val)}') # score on validation data

print(f'training score: {elasticnet.score(X_train, y_train)}') # score on training data

print(f'cross_val_score (train): {cross_val_score(elasticnet, X_train, y_train, cv=5).mean()}') # cross_val_score on training data

print(f'cross_val_score (validation): {cross_val_score(elasticnet, X_train, y_train, cv=5).mean()}') # cross_val_score on training data

# add the score to the model_scores dictionary
model_scores['elasticnet'] = elasticnet.score(X_val, y_val)
# add the predictions to the model_predictions dictionary
model_predictions['elasticnet'] = y_val_preds


train_score = elasticnet.score(X_train, y_train) # score on training data
test_score = elasticnet.score(X_val, y_val) # score on validation data
cval_score = cross_val_score(elasticnet, X_train, y_train, cv=5).mean() # score on cross validation data


# Plot the predictions vs the actual values

figure_number = 23

plt.figure(figsize=(10, 10))

plt.scatter(y_val, y_val_preds) # plot the predictions vs the actual values

plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, ls="--", c=".3")

plt.xlabel('Actual Sale Price')

plt.ylabel('Predicted Sale Price')
plt.annotate(f'model: {model} \ntrain:{train_score}\ntest:{test_score}\ncval:{cval_score}', xy=(0.05, 0.75), xycoords='axes fraction')
plt.title(f'Figure {figure_number}: {model} Predicted vs Actual Sale Price', fontsize=15)

plt.savefig(f'../images/figure_no_{figure_number}_{model}_actual_vs_predicted.png')

elasticnet_score = elasticnet.score(X_val, y_val)
plt.show();


# %%
row['model_name'] = 'ElasticNet'
row['r2_score'] = elasticnet_score
row['rmse_score'] = np.sqrt(mean_squared_error(y_val, y_val_preds))
row['train_score'] = train_score
row['test_score'] = test_score
row['cval_score'] = cval_score
row = pd.DataFrame(row, index=[0])
model_scores_df = model_scores_df.append(row, ignore_index=True)

# %%
# plot the residuals
figure_number = 24
figure = plt.figure(figsize=(10, 10))
sns.residplot(y_val_preds, y_val, lowess=True, color="g")
plt.title(f'Figure {figure_number}: ElasticNet Residuals', fontsize=15)
plt.annotate(f'model: {model} \ntrain:{train_score}\ntest:{test_score}\ncval:{cval_score}', xy=(0.05, 0.75), xycoords='axes fraction')
plt.ylabel('Residuals')
plt.xlabel('Predicted Sale Price')
plt.savefig(f'../images/figure_no_{figure_number}_{model}.png')
plt.show();


# %%
# get the coefficients of the model as a dataframe
coefficients = pd.DataFrame(elasticnet.coef_, X.columns, columns=["Coefficients"])

# Inference variables
coefficients = coefficients # set to linear regression

largest_coef_value = coefficients['Coefficients'].sort_values(ascending=False)[0]
largest_coef_feature = coefficients['Coefficients'].sort_values(ascending=False).index[0]
smallest_coef_value = coefficients['Coefficients'].sort_values(ascending=False)[-1]
smallest_coef_feature = coefficients['Coefficients'].sort_values(ascending=False).index[-1]

# Inference on the coefficients
print(f'A one unit increase in {largest_coef_feature}, holding all other features constant, will result in a {coefficients.loc[largest_coef_feature, "Coefficients"]} log-unit increase in saleprice.') # log unit increase, because we are using regularized regression

print(f'A one unit decrease in {smallest_coef_feature}, holding all other features constant, will result in a {coefficients.loc[smallest_coef_feature, "Coefficients"]} log-unit increase in saleprice.')

largest_coefs_four_elasticnet = coefficients['Coefficients'].sort_values(ascending=False)[0:4]
smallest_coefs_four_elasticnet = coefficients['Coefficients'].sort_values(ascending=False)[-4:]


# %% [markdown]
# Differences between Lasso and Ridge Models:
# * Ridge: It includes all (or none) of the features in the model. Thus, the major advantage of ridge regression is coefficient shrinkage and reducing model complexity. It is majorly used to prevent overfitting. Since it includes all the features, it is not very useful in case of exorbitantly high #features, say in millions, as it will pose computational challenges.
# * Lasso: Along with shrinking coefficients, lasso performs feature selection as well. (Remember the ‘selection‘ in the lasso full-form?) As we observed earlier, some of the coefficients become exactly zero, which is equivalent to the particular feature being excluded from the model. Since it provides sparse solutions, it is generally the model of choice (or some variant of this concept) for modelling cases where the #features are in millions or more. In such a case, getting a sparse solution is of great computational advantage as the features with zero coefficients can simply be ignored.
#
# source: https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/

# %% [markdown]
# Now, I want to display the most impactful features in the models that I have created after evaluating which of the models is the best fit for our scenario where we have a R2 goal of at least 0.8 and a RMSE goal of less than 40,000.

# %%
# comparing the coefficient values for LR, LASSO, RIDGE, and ELASTICNET
# Figure Description: This seaborn heatmap shows the coefficient values for each of the models, and the features as the x and y axis labels.
figure_number = 25
figure = plt.figure(figsize=(10, 10))
sns.heatmap(pd.DataFrame({'LR': lr.coef_, 'LASSO': lasso.coef_, 'RIDGE': ridge.coef_, 'ELASTICNET': elasticnet.coef_}, index=X.columns), annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, linecolor='black')
plt.title(f'Figure {figure_number}: Coefficient Values for LR, LASSO, RIDGE, and ELASTICNET')
plt.savefig(f'../images/figure_no_{figure_number}_coefficient_values_for_LR_LASSO_RIDGE_and_ELASTICNET.png')
plt.xlabel('Models')
plt.ylabel('Features')
plt.show();


# %% [markdown]
# When interpreting this figure, consider that Lasso, Ridge, and ElasticNet are all regularized models. This means that they are penalized for having too many features. When we interpret these coefficients we say that changes in the feature result in a increase or decrease in the log-odds rather than the odds themselves.
#
# ### Linear Coefficient Inference
# 1. The largest coefficient for the linear regression model is `year_built` (517.82). This means that for every year that a house exists (and ages),

# %%

data = pd.DataFrame.from_dict(model_scores, orient='index').T


# %%
data.head()

# %%
figure_number = 26
data.plot(kind='bar', figsize=(10, 5), title=f'Figure {figure_number}: Model Scores')

# %%
model_predictions = pd.DataFrame(model_predictions)
model_predictions.head()

# %%
# combine model_predictions and model_scores into a single dataframe with a added column for the model name.
model_df = pd.DataFrame()
for model in model_predictions.columns:
    row = {'model': model, 'predictions': model_predictions[model].to_list(), 'score': model_scores[m












    odel]}
    model_df = model_df.append(row, ignore_index=True)
model_df.head()

# %%
len(y_val), len(model_df['predictions'][3])

# %%
# Plot the predictions vs the actual values
figure_number = 27
model_name = 'LinearRegression'
plt.figure(figsize=(10, 10))

plt.scatter(y_val, model_df['predictions'][0], alpha = 0.5) # plot the predictions vs the actual values for the first model lr
plt.scatter(y_val, model_df['predictions'][1],alpha = 0.5) # plot the predictions vs the actual values for lasso
plt.scatter(y_val, model_df['predictions'][2],alpha = 0.5) # plot the predictions vs the actual values for ridge
plt.scatter(y_val, model_df['predictions'][3],alpha = 0.5) # plot the predictions vs the actual values for the elasticnet model
plt.plot([0, 1], [0, 1],
         transform=plt.gca().transAxes, ls="--", c=".3")
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title(f'Figure {figure_number}: {model_name} Predicted vs Actual Sale Price')
plt.legend(['lr', 'lasso', 'ridge', 'elasticnet'])
plt.savefig(f'../images/figure_no_{figure_number}_{model_name}_actual_vs_predicted.png')
plt.show();

# %%


#TODO --- lr is not using Scaled data... correct that
df = pd.DataFrame({'lr': lr.coef_, 'lasso': lasso.coef_, 'ridge': ridge.coef_, 'elasticnet': elasticnet.coef_}, index=X.columns)

figure_number = 28# plot the residuals

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(30,5))
axes[0].scatter(y_val, lr.predict(X_val_lr))
axes[1].scatter(y_val, lasso.predict(X_val))
axes[2].scatter(y_val, ridge.predict(X_val))
axes[3].scatter(y_val, elasticnet.predict(X_val))
plt.suptitle(f'Figure {figure_number}. Comparison of the 4 models',fontsize = 20)
axes[0].set_xlabel("Actual Prices") # set the x label
axes[0].set_ylabel("Predicted Prices") # set the y label
axes[0].set_title("Linear Regression") # set the title
axes[1].set_title("Lasso")
axes[1].set_xlabel("Actual Prices") # set the x label
axes[3].set_xlabel("Actual Prices") # set the x label
axes[2].set_xlabel("Actual Prices") # set the x label
axes[2].set_title("Ridge")
axes[3].set_title("ElasticNet")

axes[0].annotate(f'R^2: {round(lr.score(X_val_lr, y_val),6)}', xy=(0.1, 0.9), xycoords='axes fraction')
axes[0].annotate(f'RMSE: {round(np.sqrt(mean_squared_error(y_val, lr.predict(X_val_lr))),6)}', xy=(0.1, 0.85), xycoords='axes fraction')
axes[1].annotate(f'R^2: {round(lasso.score(X_val, y_val),6)}', xy=(0.1, 0.9), xycoords='axes fraction')
axes[1].annotate(f'RMSE: {round(np.sqrt(mean_squared_error(y_val, lasso.predict(X_val))),6)}', xy=(0.1, 0.85), xycoords='axes fraction')
axes[2].annotate(f'R^2: {round(ridge.score(X_val, y_val),6)}', xy=(0.1, 0.9), xycoords='axes fraction')
axes[2].annotate(f'RMSE: {round(np.sqrt(mean_squared_error(y_val, ridge.predict(X_val))),6)}', xy=(0.1, 0.85), xycoords='axes fraction')
axes[3].annotate(f'R^2: {round(elasticnet.score(X_val, y_val),6)}', xy=(0.1, 0.9), xycoords='axes fraction')
axes[3].annotate(f'RMSE: {round(np.sqrt(mean_squared_error(y_val, elasticnet.predict(X_val))),6)}', xy=(0.1, 0.85), xycoords='axes fraction')
plt.savefig(f"../images/figure_no_{figure_number}_basic_comparison_of_the_4_models.png")
plt.show()


# %% [markdown]
# # Conclusions and Recommendations

# %%
model_scores_df.head()

# %% [markdown]
# The results show that (based on r2 scores) the lr and lasso models are tied but lasso reduces RMSE by a very tiny percentage.
# The best RMSE score is from the Ridge model (44239.896572). We would either want to boost this model, to increase the complexity or we may want to consider adding more features into the analysis to increase the R2 score.
#
# # Conclusions and Final Remarks
# Based on the results of this regression analysis, we can conclude that the following features are the most important in predicting the sale price of a house in Ames, Iowa.
#
# * `overall_qual`
# * `gr_liv_area`
# * `garage_area`
# * `garage_cars`
# * `total_bsmt_sf`
#
# We recommend that our client take time and focus on the following features to find the ideal property to purchase.
#
# The primary features that we recommend our client focus on are `overall_qual`, `gr_liv_area`, `garage_area`, `garage_cars`, and `total_bsmt_sf`.
#
#
# This project has been quite successful. We were able to develop a predictive model using only publicly available information about a home's features. This allows us to make predictions based solely on the public record without spending any money or effort to gather more data ourselves. In addition to being cost-effective, it also provides a quick solution to identify properties that may be worth purchasing. Our model shows some promise; however, we would like to investigate whether there is anything else that could improve its performance. For example, perhaps we can use machine learning techniques such as boosting to predict the sale price better.
# To continue improving our model, we plan to run the same regression analysis on additional data sets. Ames rental data is an exciting area for further study, and we are considering recommending this to our clients as a way to see an ROI on their investment quicker than with the purchase market. Additionally, we believe that adding additional features to our model might help us achieve even higher accuracy rates.
#
#
#

# %% [markdown]
#

# %% [markdown]
# # References

# %% [markdown]
# [1] De Cock, Dean. "Ames, Iowa: Alternative to the Boston housing data as an end of semester regression project." Journal of Statistics Education 19.3 (2011).
#
# [2] https://www.python-graph-gallery.com/web-ggbetweenstats-with-matplotlib
#
