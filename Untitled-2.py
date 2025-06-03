# %%
# Import essential libraries for data manipulation, visualization, and modeling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import scikit-learn modules for preprocessing, modeling, and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_squared_error, r2_score, mean_absolute_error)

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# %%
# Load dataset
df = pd.read_csv("Influenza_NY.csv")

# Display first 5 rows to get an overview of the data
print(df.head())

# %%
# Check the shape (rows, columns) of the dataset
print("Shape:", df.shape)

# %%
# Display data types of each column to understand variable types
print("Data Types:\n", df.dtypes)

# %%
# Check for missing values in each column
print("Missing Values:\n", df.isnull().sum())

# %%
# Explore unique disease types in the dataset
print("Unique Diseases:", df["Disease"].unique())

# %%
# Scale numerical features for normalization (mean=0, std=1)
numerical_cols = ['Avg household size', 'Area', 'Population', 'Median_age', 'Medianfamilyincome',
                  'Number_households', 'Beds_adult_facility_care', 'Beds_hospital',
                  'Discharges_Other_Hospital_intervention', 'Discharges_Respiratory_system_interventions',
                  'Total_Charge_Other_Hospital_intervention', 'Total_Charge_Respiratory_system_interventions', 
                  'Unemp_rate']

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# %%
# Visualize distribution of infected cases using histogram with KDE
plt.figure(figsize=(10, 6))
sns.histplot(df['Infected'], bins=30, kde=True)
plt.title("Distribution of Infected Cases")
plt.xlabel("Infected Cases")
plt.ylabel("Frequency")
plt.show()

# %%
# Create 'YearMonth' for time series analysis and plot monthly infection trends
df['YearMonth'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)

monthly_infected = df.groupby('YearMonth')['Infected'].sum().reset_index()

plt.figure(figsize=(18, 9))
sns.lineplot(data=monthly_infected, x='YearMonth', y='Infected', marker='o')
plt.xticks(rotation=45)
plt.title("Monthly Infection Trend Over Time")
plt.xlabel("Year-Month")
plt.ylabel("Total Infected Cases")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Visualize correlation heatmap between numerical features and the target variable
plt.figure(figsize=(14, 10))
correlation = df[numerical_cols + ['Infected']].corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# %%
# Drop rows where target variable 'Infected' is missing
df = df.dropna(subset=["Infected"])

# %%
# Remove outliers from the target variable using the IQR method
Q1 = df['Infected'].quantile(0.25)
Q3 = df['Infected'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['Infected'] >= lower_bound) & (df['Infected'] <= upper_bound)]

# %%
# Re-scale numerical features after outlier removal
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# %%
# Convert 'YearMonth' back to datetime and extract 'Year' and 'Month' features
df['YearMonth'] = pd.to_datetime(df['YearMonth'], format='%Y-%m')
df['Year'] = df['YearMonth'].dt.year
df['Month'] = df['YearMonth'].dt.month
df.drop(columns=['YearMonth'], inplace=True)

# %%
# One-hot encode categorical features for modeling
df = pd.get_dummies(df, columns=["Disease", "Region", "Season", "County"])

# %%
# Define feature matrix X and target vector y
X = df.drop(columns=["Infected", "Week Ending Date"])  # Drop target and irrelevant columns
y = df["Infected"]

# %%
# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Handle missing values in numerical features with mean imputation
numeric_cols = X.select_dtypes(include=['number']).columns

imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train[numeric_cols]), columns=numeric_cols, index=X_train.index)
X_test_imputed = pd.DataFrame(imputer.transform(X_test[numeric_cols]), columns=numeric_cols, index=X_test.index)

# %%
# Scale numerical features after imputation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# %%
# Dimensionality Reduction: Apply PCA to visualize data in 2D space
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='coolwarm', alpha=0.7)
plt.title("PCA - First 2 Principal Components")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Infected Cases")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# MODEL TRAINING & EVALUATION

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Print regression metrics for evaluation."""
    print(f"{model_name} Performance:")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print("-" * 30)

# %%
# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)
evaluate_model(y_test, y_pred_dt, "Decision Tree Regressor")

# %%
# Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
evaluate_model(y_test, y_pred_rf, "Random Forest Regressor")

# %%
# Support Vector Machine Regressor
svm = SVR()
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
evaluate_model(y_test, y_pred_svm, "Support Vector Machine Regressor")

# %%
# K-Nearest Neighbors Regressor with pipeline including imputation
knn_pipeline = make_pipeline(SimpleImputer(strategy='mean'), KNeighborsRegressor())
knn_pipeline.fit(X_train_scaled, y_train)
y_pred_knn = knn_pipeline.predict(X_test_scaled)
evaluate_model(y_test, y_pred_knn, "K-Nearest Neighbors Regressor")

# %%
# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)
y_pred_lr = linear_reg.predict(X_test_scaled)
evaluate_model(y_test, y_pred_lr, "Linear Regression")

# %%
# Logistic Regression for classification (optional)
# Here we binarize the target for logistic regression demonstration
y_binary = (y > y.median()).astype(int)
y_train_bin = y_binary.loc[y_train.index]
y_test_bin = y_binary.loc[y_test.index]

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train_bin)
y_pred_log_reg = log_reg.predict(X_test_scaled)

print("Logistic Regression Classification Report:")
print(f"Accuracy: {accuracy_score(y_test_bin, y_pred_log_reg):.4f}")
print(classification_report(y_test_bin, y_pred_log_reg))
print("Confusion Matrix:\n", confusion_matrix(y_test_bin, y_pred_log_reg))
