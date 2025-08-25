# California House Price Prediction
# Alternative implementation since Boston housing dataset is deprecated

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Load the California housing dataset
print("Loading California Housing Dataset...")
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

# Create DataFrame for better visualization
feature_names = california_housing.feature_names
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"Dataset shape: {df.shape}")
print("\nFeature names:")
for i, name in enumerate(feature_names):
    print(f"{i + 1}. {name}")

print(f"\nTarget variable: House Value (in hundreds of thousands of dollars)")
print(f"Target range: ${y.min():.2f} - ${y.max():.2f} (hundreds of thousands)")

# Display basic statistics
print("\n" + "=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(df.describe())

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# Exploratory Data Analysis
print("\n" + "=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# Set up the plotting style
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Distribution of target variable
axes[0, 0].hist(y, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of House Prices')
axes[0, 0].set_xlabel('Price (hundreds of thousands $)')
axes[0, 0].set_ylabel('Frequency')

# Correlation heatmap
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            ax=axes[0, 1], fmt='.2f', square=True)
axes[0, 1].set_title('Feature Correlation Matrix')

# Scatter plot: MedInc vs Target
axes[1, 0].scatter(df['MedInc'], df['target'], alpha=0.5)
axes[1, 0].set_xlabel('Median Income')
axes[1, 0].set_ylabel('House Price')
axes[1, 0].set_title('Median Income vs House Price')

# Feature importance visualization (using correlation with target)
feature_importance = correlation_matrix['target'].drop('target').abs().sort_values(ascending=False)
axes[1, 1].barh(range(len(feature_importance)), feature_importance.values)
axes[1, 1].set_yticks(range(len(feature_importance)))
axes[1, 1].set_yticklabels(feature_importance.index)
axes[1, 1].set_xlabel('Absolute Correlation with Target')
axes[1, 1].set_title('Feature Importance (Correlation)')

plt.tight_layout()
plt.show()

# Prepare data for modeling
print("\n" + "=" * 50)
print("DATA PREPARATION")
print("=" * 50)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training and Evaluation
print("\n" + "=" * 50)
print("MODEL TRAINING AND EVALUATION")
print("=" * 50)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Use scaled data for linear models, original data for Random Forest
    if name == 'Random Forest':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'predictions': y_pred,
        'model': model
    }

    print(f"{name} Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")

# Results Comparison
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    name: [results[name]['MSE'], results[name]['RMSE'],
           results[name]['MAE'], results[name]['R²']]
    for name in models.keys()
}, index=['MSE', 'RMSE', 'MAE', 'R²'])

print(comparison_df.round(4))

# Find best model
best_model_name = min(results.keys(), key=lambda x: results[x]['RMSE'])
print(f"\nBest Model (lowest RMSE): {best_model_name}")

# Visualization of Results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Model comparison
metrics = ['MSE', 'RMSE', 'MAE', 'R²']
colors = ['red', 'blue', 'green', 'orange']

for i, metric in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    values = [results[name][metric] for name in models.keys()]
    bars = ax.bar(models.keys(), values, color=colors[i], alpha=0.7)
    ax.set_title(f'{metric} Comparison')
    ax.set_ylabel(metric)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Prediction vs Actual for best model
plt.figure(figsize=(10, 6))
best_predictions = results[best_model_name]['predictions']

plt.scatter(y_test, best_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title(f'Actual vs Predicted Prices - {best_model_name}')
plt.tight_layout()
plt.show()

# Feature importance for Random Forest
if 'Random Forest' in results:
    plt.figure(figsize=(10, 6))
    rf_model = results['Random Forest']['model']
    feature_importance = rf_model.feature_importances_

    sorted_idx = np.argsort(feature_importance)[::-1]

    plt.bar(range(len(feature_importance)), feature_importance[sorted_idx])
    plt.xticks(range(len(feature_importance)),
               [feature_names[i] for i in sorted_idx], rotation=45, ha='right')
    plt.title('Feature Importance - Random Forest')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()

# Sample predictions
print("\n" + "=" * 50)
print("SAMPLE PREDICTIONS")
print("=" * 50)

# Show some sample predictions with the best model
best_model = results[best_model_name]['model']
n_samples = 5

print(f"Using {best_model_name} for predictions:")
print("Actual vs Predicted (first 5 test samples):")
print("-" * 40)

for i in range(n_samples):
    actual = y_test[i]
    predicted = best_predictions[i]
    error = abs(actual - predicted)
    print(f"Sample {i + 1}: Actual=${actual:.2f}00k, Predicted=${predicted:.2f}00k, Error=${error:.2f}00k")


# Function to make predictions on new data
def predict_house_price(med_inc, house_age, avg_rooms, avg_bedrooms, population, avg_occupancy, latitude, longitude):
    """
    Predict house price given house features
    """
    # Create feature array
    features = np.array([[med_inc, house_age, avg_rooms, avg_bedrooms,
                          population, avg_occupancy, latitude, longitude]])

    # Scale features if using linear models
    if best_model_name != 'Random Forest':
        features = scaler.transform(features)

    # Make prediction
    prediction = best_model.predict(features)[0]
    return prediction * 100  # Convert to thousands


print(f"\n" + "=" * 50)
print("EXAMPLE PREDICTION")
print("=" * 50)

# Example prediction
example_price = predict_house_price(
    med_inc=5.0,  # Median income
    house_age=10.0,  # House age
    avg_rooms=6.0,  # Average rooms
    avg_bedrooms=1.2,  # Average bedrooms
    population=3000,  # Population
    avg_occupancy=3.0,  # Average occupancy
    latitude=34.0,  # Latitude
    longitude=-118.0  # Longitude
)

print(f"Example prediction: ${example_price:.2f}k")
print("\nFeature explanations:")
print("- MedInc: Median income in block group")
print("- HouseAge: Median house age in block group")
print("- AveRooms: Average number of rooms per household")
print("- AveBedrms: Average number of bedrooms per household")
print("- Population: Block group population")
print("- AveOccup: Average number of household members")
print("- Latitude: Block group latitude")
print("- Longitude: Block group longitude")

print(f"\n" + "=" * 50)
print("MODEL SUMMARY")
print("=" * 50)
print(f"Best performing model: {best_model_name}")
print(f"R² Score: {results[best_model_name]['R²']:.4f}")
print(f"RMSE: {results[best_model_name]['RMSE']:.4f} (hundreds of thousands $)")
print(
    f"This means the model can predict house prices with an average error of ${results[best_model_name]['RMSE'] * 100:.0f}k")