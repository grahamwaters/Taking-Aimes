load the data
data = pd.read_csv('ames_housing_data.csv')

view the first few rows of the data
data.head()

check the dimensions of the data
data.shape

check the data types
data.dtypes

check for missing values
data.isnull().sum()

calculate summary statistics for the numerical variables
data.describe()

visualize the distribution of the target variable, SalePrice
sns.distplot(data['SalePrice'])

visualize the relationship between SalePrice and overall_qual
sns.scatterplot(x='overall_qual', y='SalePrice', data=data)

compute the correlation matrix
corr = data.corr()

visualize the correlation matrix using a heatmap
sns.heatmap(corr)

split the data into training and testing sets
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

compute the variance inflation factors (VIF)
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

view the VIF values
vif

fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

evaluate the model using cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-validation scores:", scores)
print("Average cross-validation score:", scores.mean())

make predictions on the testing set
y_pred = model.predict(X_test)

evaluate the model using the mean squared error (MSE) and R2 scores
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

fit elastic net and ridge regression models
enet = ElasticNet()
ridge = Ridge()

evaluate the models using cross-validation
enet_scores = cross_val_score(enet, X_train, y_train, cv=5)
ridge_scores = cross_val_score(ridge, X_train, y_train, cv=5)

view the cross-validation scores
print("Elastic net cross-validation scores:", enet_scores)
print("Ridge cross-validation scores:", ridge_scores)

fit a LassoCV model
lasso = LassoCV()
lasso.fit(X_train, y_train)

evaluate the model using cross-validation
lasso_scores = cross_val_score(lasso, X_train, y_train, cv=5)
print("LassoCV cross-validation scores:", lasso

Compare the performance of the LassoCV model
compute the mean and standard deviation of the cross-validation scores for each model
linear_mean, linear_std = np.mean(scores), np.std(scores)
enet_mean, enet_std = np.mean(enet_scores), np.std(enet_scores)
ridge_mean, ridge_std = np.mean(ridge_scores), np.std(ridge_scores)
lasso_mean, lasso_std = np.mean(lasso_scores), np.std(lasso_scores)

view the cross-validation scores for each model
print("Linear regression cross-validation scores: Mean = %0.3f, Std = %0.3f" % (linear_mean, linear_std))
print("Elastic net cross-validation scores: Mean = %0.3f, Std = %0.3f" % (enet_mean, enet_std))
print("Ridge cross-validation scores: Mean = %0.3f, Std = %0.3f" % (ridge_mean, ridge_std))
print("LassoCV cross-validation scores: Mean = %0.3f, Std = %0.3f" % (lasso_mean, lasso_std))

evaluate the models on the testing set
linear_mse = mean_squared_error(y_test, model.predict(X_test))
enet_mse = mean_squared_error(y_test, enet.predict(X_test))
ridge_mse = mean_squared_error(y_test, ridge.predict(X_test))
lasso_mse = mean_squared_error(y_test, lasso.predict(X_test))

view the MSE scores for each model
print("Linear regression MSE:", linear_mse)
print("Elastic net MSE:", enet_mse)
print("Ridge MSE:", ridge_mse)
print("LassoCV MSE:", lasso_mse)

evaluate the models on the testing set
linear_r2 = r2_score(y_test, model.predict(X_test))
enet_r2 = r2_score(y_test, enet.predict(X_test))
ridge_r2 = r2_score(y_test, ridge.predict(X_test))
lasso_r2 = r2_score(y_test, lasso.predict(X_test))

view the R2 scores for each model
