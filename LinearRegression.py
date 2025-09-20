import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import  KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

# Function to preprocess the dataset (for 2 datasets)
def preprocess_data(df):

    # change the date coloumn using datetime since when using .info() date was considered object
    df['date'] = pd.to_datetime(df['date'])

    # extract year, month, dayofweek, and quarter from date
    df['sale_year'] = df['date'].dt.year
    df['sale_month'] = df['date'].dt.month
    df['sale_dayofweek'] = df['date'].dt.dayofweek
    df['sale_quarter'] = df['date'].dt.quarter

    # no longer need the date coloumn
    df = df.drop(columns=['date'])

    # Convert 'sale_dayofweek' to dummy variables & convert bool types to int type
    df = pd.get_dummies(df, columns=['sale_dayofweek'], drop_first=True)

    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # no need for price coloumn since we are predicting it, drop it
    df = df.dropna(subset=['price'])

    # Separate numeric and non-numeric columns its nessisary since after scalar transfrom,
    # some of the orignial boolean data types will be converted into float64 type
    continuous_cols = ['price', 'bedrooms', 'grade', 'living_in_m2', 'real_bathrooms', 
                       'month', 'quartile_zone', 'sale_year', 'sale_month', 'sale_quarter']
    binary_cols = [col for col in df.columns if col not in continuous_cols and col != 'price']

    # fill any missing values with median value of that coloumn
    df[continuous_cols] = df[continuous_cols].fillna(df[continuous_cols].median())

    # Scale only the continuous numeric columns aka floating type variables
    scaler = StandardScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    return df

# Load and preprocess the training data
df_train = pd.read_csv('df_train.csv')
df_train = preprocess_data(df_train)

# print(df_train.info()) # check all feature data types withing dataframe
# print(df_train)        # take a look at dataframe

# Separate features and target variable in training data
X_train = df_train.drop('price', axis=1)
y_train = df_train['price']

# Load and preprocess the testing data
df_test = pd.read_csv('df_test.csv')
df_test = preprocess_data(df_test)

# Ensure the test set has the same features as the training set
X_test = df_test.drop('price', axis=1)
y_test = df_test['price']

# Select top 10 features that contribute the most to price for a linear regression model
selector = SelectKBest(score_func=f_regression, k=15)
# maintain consistent amount of features betweeen train and test data sets
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# train the current model on the 15 featrures selected earlier using slectkbest(house price)
model = LinearRegression()
model.fit(X_train_selected, y_train)

# predcit on test data
y_pred = model.predict(X_test_selected)
# get root mean squared error
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Root Mean Squared Error on Test Set: {rmse}")


# scatter plot for actual vs predicted prices
plt.figure(figsize=(15, 11))
# plot each pair of actual and predicted prices
sns.scatterplot(x=y_test, y=y_pred)
# plots ideal line for perfect prediciton
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual prices vs Predcites prices of houses')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# Get the columns selected from data frame
selected_features = X_train.columns[selector.get_support()]
# print("Selected Features:", selected_features)

# create a data frame to store the folded features distances and rmse for each
results_df = pd.DataFrame(columns=['Fold'] + list(selected_features) + ['RMSE'])

# Perform 10-fold cross-validation on the training data
# shuffle set to true since it can improve generalization, same each time '42'
kf = KFold(n_splits=10, shuffle=True, random_state=42)
# loop and place in data frame for each fold providing train and val index to acess the subsets
for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train_selected), 1):
    # Split the data for the current fold
    X_train_fold, X_val_fold = X_train_selected[train_index], X_train_selected[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # retrain the model on the current fold
    model = LinearRegression()
    model.fit(X_train_fold, y_train_fold)
    
    # repredict and calculate RMSE on the fold
    y_val_pred = model.predict(X_val_fold)
    rmse = root_mean_squared_error(y_val_fold, y_val_pred)
    
    # store the rmse and the coefficient intot he data frame
    fold_results = [fold_idx] + list(model.coef_) + [rmse]
    results_df.loc[fold_idx - 1] = fold_results

# output the average row, and the mean of the rmse at the last index
average_rmse = results_df['RMSE'].mean()
average_row = ['Average'] + list(results_df.iloc[:, 1:-1].mean()) + [average_rmse]
results_df.loc[len(results_df)] = average_row

df_subset = results_df.iloc[:, 6:]

print(df_subset)
print(results_df)
