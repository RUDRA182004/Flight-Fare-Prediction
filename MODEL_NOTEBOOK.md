# Model.ipynb — Machine Learning Pipeline Documentation

This document explains the Jupyter notebook `Model.ipynb`, which contains the complete machine learning workflow for training the flight fare prediction model.

## Overview

The notebook builds a Random Forest regression model to predict flight fares from flight booking attributes. It uses scikit-learn for modeling, pandas for data manipulation, and matplotlib/seaborn for visualization.

---

## Dataset

### Input file: `a1_FlightFare_Dataset.xlsx`

The training dataset contains flight booking records from Indian airlines with the following original columns:

| Column | Data Type | Description |
|--------|-----------|-------------|
| `Airline` | Categorical | Airline operator (e.g., Jet Airways, IndiGo, Air India, SpiceJet, Multiple carriers, GoAir, Vistara, Air Asia, others) |
| `Date_of_Journey` | String | Date of departure in `DD/MM/YYYY` format |
| `Source` | Categorical | Departure city (Delhi, Kolkata, Mumbai, Chennai, Banglore) |
| `Destination` | Categorical | Arrival city (Cochin, Delhi, New Delhi, Hyderabad, Kolkata) |
| `Route` | Categorical | Intermediate stops/routing information (later dropped as redundant) |
| `Dep_Time` | String | Departure time in `HH:MM` format (24-hour) |
| `Arrival_Time` | String | Arrival time in `HH:MM` format (24-hour) |
| `Duration` | String | Flight duration in `XhYm` format (e.g., "2h 30m", "19h", "45m") |
| `Total_Stops` | Categorical | Number of stops (non-stop, 1 stop, 2 stops, 3 stops, 4 stops) |
| `Additional_Info` | Categorical | Additional booking information (mostly "No Info", ~80% missing; later dropped) |
| `Price` | Numeric (target) | Flight ticket price in Indian Rupees (INR) — **target variable** |

### Dataset characteristics

- **Size**: Multiple thousands of booking records (exact count varies after cleaning)
- **Time period**: Flight bookings for Indian domestic routes
- **Missing values**: Minimal (only 2 rows dropped)
- **Data quality**: Generally clean; duplication handled via feature engineering

### Unseen validation dataset: `a2_Unseen_Dataset.xlsx`

- Contains the same columns as the training dataset
- Used to validate model performance on held-out data
- Predictions written to `c2_ModelOutput.xlsx`

---

## Pipeline stages

### 1. Setup and data loading

**Cell 1–5**: Install and import dependencies
- Installs `scikit-learn`, `pandas`, `numpy`
- Imports: `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `sklearn.metrics`

**Cell 6**: Load training dataset
```python
df = pd.read_excel('a1_FlightFare_Dataset.xlsx')
df.head()
```
- Loads the training data from an Excel file
- Data inspection shows rows and columns

**Cell 7–9**: Null value check and removal
```python
df.info()  # Check data types and non-null counts
df.isnull().sum()  # Count missing values
df.dropna(inplace=True)  # Remove rows with NaN
```
- Only two rows have missing values; they are dropped
- No scaling or imputation needed

---

### 2. Feature engineering

#### 2.1 Datetime features

**Cell 10–12**: Extract date features from `Date_of_Journey`
```python
df["journey_day"] = pd.to_datetime(df.Date_of_Journey, format="%d/%m/%Y").dt.day
df["journey_month"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.month
df.drop("Date_of_Journey", axis=1, inplace=True)
```
- Extracts day-of-month and month as separate numeric columns
- Drops original date column

**Cell 13–15**: Extract time features from `Dep_Time`
```python
df["dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
df["dep_min"] = pd.to_datetime(df["Dep_Time"]).dt.minute
df.drop(["Dep_Time"], axis=1, inplace=True)
```
- Extracts departure hour and minute
- Drops original time column

**Cell 16–18**: Extract time features from `Arrival_Time`
```python
df["arrival_hour"] = pd.to_datetime(df["Arrival_Time"]).dt.hour
df["arrival_min"] = pd.to_datetime(df["Arrival_Time"]).dt.minute
df.drop(["Arrival_Time"], axis=1, inplace=True)
```
- Extracts arrival hour and minute
- Drops original time column

**Cell 19–21**: Parse and extract `Duration` features
```python
duration = list(df["Duration"])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:  # Handle missing hours or minutes
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i]
```
- Handles both "X hours Y minutes" and incomplete formats (e.g., "19h" or "30m")
- Splits into `Duration_hours` and `Duration_mins` columns
- Drops original `Duration` column

#### 2.2 Categorical features (one-hot encoding)

**Cell 22–24**: Encode `Airline`
```python
Airline = df[["Airline"]]
# Group infrequent airlines into 'Other'
Airline = pd.get_dummies(Airline, drop_first=True).astype(int)
```
- One-hot encodes airline; less frequent ones grouped as "Other"
- `drop_first=True` prevents multicollinearity
- Creates binary columns: `Airline_Air India`, `Airline_GoAir`, `Airline_IndiGo`, etc.

**Cell 25–27**: Encode `Source`
```python
Source = df[["Source"]]
Source = pd.get_dummies(Source, drop_first=True).astype(int)
```
- One-hot encodes source cities (Delhi, Kolkata, Mumbai, Chennai)
- `drop_first=True` prevents the dummy variable trap

**Cell 28–30**: Encode and normalize `Destination`
```python
Destination = df[["Destination"]]
# Rename 'New Delhi' to 'Delhi' for consistency with Source
Destination = pd.get_dummies(Destination, drop_first=True).astype(int)
```
- Normalizes destination names (e.g., "New Delhi" → "Delhi")
- One-hot encodes (Cochin, Delhi, Hyderabad, Kolkata)

**Cell 31**: Drop irrelevant columns
```python
df.drop(["Route", "Additional_Info"], axis=1, inplace=True)
```
- `Route` is redundant with Source and Destination
- `Additional_Info` is ~80% missing values

#### 2.3 Ordinal categorical feature

**Cell 32–34**: Encode `Total_Stops` using label encoding
```python
df.replace({
    "non-stop": 0, "1 stop": 1, "2 stops": 2, 
    "3 stops": 3, "4 stops": 4
}, inplace=True)
```
- Label encodes stops as ordinal numeric values (0–4)
- Suitable because stops have a natural order

**Cell 35–37**: Combine and finalize features
```python
data_train = pd.concat([df, Airline, Source, Destination], axis=1)
data_train.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)
```
- Combines numeric and encoded categorical columns
- Removes original categorical columns

---

### 3. Feature selection

**Cell 38–40**: Inspect final feature set
```python
data_train.columns
X = data_train.loc[:, ['Total_Stops', 'journey_day', ...]]  # 25 features
y = data_train.iloc[:, 1]  # Price (target)
```
- Feature matrix `X` with 25 numeric features
- Target vector `y` (flight price)

**Cell 41–43**: Compute feature importances
```python
from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)
print(selection.feature_importances_)
```
- Uses ExtraTreesRegressor to rank feature importance
- Visualizes top 25 features in a bar plot

#### Multicollinearity check (VIF)

**Cell 44–46**: Calculate Variance Inflation Factor (VIF)
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
```
- Identifies highly correlated features
- High VIF (>5–10) suggests redundancy

**Cell 47**: Drop `Source_Delhi` to reduce multicollinearity
```python
X = data_train.loc[:, ['Total_Stops', ..., 'Source_Chennai', 'Source_Kolkata', 'Source_Mumbai', ...]]
# Final 24 features (Source_Delhi removed)
```
- One source city must be dropped to prevent perfect multicollinearity

---

### 4. Model training

**Cell 48–50**: Train-test split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- 80% training, 20% testing
- `random_state=42` ensures reproducibility

**Cell 51–53**: Fit Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)
```
- Default hyperparameters used
- No scaling needed for tree-based models

---

### 5. Model evaluation

**Cell 54–56**: Performance metrics
```python
print('Train R² Score:', round(rf_reg.score(X_train, y_train)*100, 2))
print('Test R² Score:', round(rf_reg.score(X_test, y_test)*100, 2))
```
- R² on training set (typically higher due to overfitting)
- R² on test set (more realistic generalization metric)

**Cell 57–59**: Prediction scatter plot
```python
y_pred = rf_reg.predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()
```
- Visualizes actual vs. predicted prices
- Good predictions cluster near the diagonal

**Cell 60–64**: Error metrics
```python
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Normalized RMSE:', round(np.sqrt(MSE) / (max(y_test) - min(y_test)), 2))
```
- **MAE**: Mean absolute error in rupees
- **RMSE**: Root mean squared error (penalizes large errors)
- **Normalized RMSE**: RMSE divided by price range (0–1 scale)

---

### 6. Model serialization

**Cell 65–68**: Save model to pickle file
```python
import pickle
file = open('c1_flight_rf_new.pkl', 'wb')
pickle.dump(rf_reg, file)
```
- Saves trained model to `c1_flight_rf_new.pkl`
- File is loaded by `app.py` for predictions

---

### 7. Validation on unseen data

**Cell 69–71**: Load and apply feature engineering to unseen data
```python
unseen_dataset = pd.read_excel('a2_Unseen_Dataset.xlsx')
# Apply identical feature engineering steps
# Date extraction, duration parsing, categorical encoding, multicollinearity fix
```
- Loads second dataset from Excel
- Applies **exact same** feature engineering pipeline
- Ensures consistency between training and inference

**Cell 72–74**: Extract features and generate predictions
```python
X_unseen = unseen_dataset.loc[:, [...]]  # 24 features (matches training)
y_pred = rf_model.predict(X_unseen)
```
- Loads previously saved model
- Generates predictions on unseen test set

**Cell 75–77**: Evaluate on unseen data
```python
print('Normalized RMSE:', round(np.sqrt(metrics.mean_squared_error(y_unseen, y_pred)) / (max(y_unseen) - min(y_unseen)), 2))
print('R² value:', round(metrics.r2_score(y_unseen, y_pred), 2))
```
- Compares performance on a separate held-out test set
- Validates model generalization

**Cell 78–80**: Write output to Excel
```python
df_y_pred = pd.DataFrame(y_pred, columns=['Predicted Price'])
original_dataset = pd.read_excel("./a2_Unseen_Dataset.xlsx")
dfx = pd.concat([original_dataset, df_y_pred], axis=1)
dfx.to_excel("c2_ModelOutput.xlsx")
```
- Combines original unseen data with predictions
- Saves to `c2_ModelOutput.xlsx`

---

### 8. Dependencies and versions

**Cell 81–85**: Print versions
```python
print("sklearn:", sklearn.__version__)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
```
- Confirms installed package versions for reproducibility

---

## Key features of the pipeline

| Feature | Details |
|---------|---------|
| **Input dataset** | Excel file: `a1_FlightFare_Dataset.xlsx` |
| **Target variable** | Flight price (regression) |
| **Algorithm** | Random Forest Regressor |
| **Feature count** | 24 numeric features (after multicollinearity removal) |
| **Train/test split** | 80/20 |
| **Model file** | `c1_flight_rf_new.pkl` |
| **Evaluation metric** | R², MAE, RMSE |
| **Validation dataset** | `a2_Unseen_Dataset.xlsx` → `c2_ModelOutput.xlsx` |

---

## How to run the notebook

1. Ensure `a1_FlightFare_Dataset.xlsx` is in the project root
2. Open `Model.ipynb` in Jupyter Lab or VS Code
3. Run all cells in order
4. Check outputs:
   - Model R² score
   - Feature importance plot
   - RMSE and error metrics
   - Generated `c1_flight_rf_new.pkl`
   - Generated `c2_ModelOutput.xlsx` (if unseen data provided)

---

## Notes

---

## Final feature set and transformations

### Original columns → Engineered features mapping

| Original Column | Engineered Features | Transform Type | Notes |
|-----------------|-------------------|-----------------|-------|
| `Date_of_Journey` | `journey_day`, `journey_month` | Datetime extraction | Day-of-month (1–31) and month (1–12) |
| `Dep_Time` | `dep_hour`, `dep_min` | Datetime extraction | Hour (0–23) and minute (0–59) |
| `Arrival_Time` | `arrival_hour`, `arrival_min` | Datetime extraction | Hour (0–23) and minute (0–59) |
| `Duration` | `Duration_hours`, `Duration_mins` | String parsing | Handles "XhYm", "Xh", "Ym" formats; defaults missing hour/min to 0 |
| `Total_Stops` | `Total_Stops` (numeric) | Label encoding | 0 (non-stop), 1–4 (number of stops) |
| `Airline` | 8 one-hot columns | One-hot encoding | `Airline_Air India`, `Airline_GoAir`, `Airline_IndiGo`, `Airline_Jet Airways`, `Airline_Multiple carriers`, `Airline_Other`, `Airline_SpiceJet`, `Airline_Vistara` |
| `Source` | 4 one-hot columns | One-hot encoding (drop_first=True) | `Source_Chennai`, `Source_Kolkata`, `Source_Mumbai` (Delhi dropped to prevent multicollinearity) |
| `Destination` | 4 one-hot columns | One-hot encoding (drop_first=True) | `Destination_Cochin`, `Destination_Delhi`, `Destination_Hyderabad`, `Destination_Kolkata` |
| `Route` | — | Dropped | Redundant with Source + Destination |
| `Additional_Info` | — | Dropped | ~80% missing values; negligible information |

### Final model input: 24 numeric features

```
[Total_Stops, journey_day, journey_month, dep_hour, dep_min, arrival_hour, 
arrival_min, Duration_hours, Duration_mins, Airline_Air India, Airline_GoAir, 
Airline_IndiGo, Airline_Jet Airways, Airline_Multiple carriers, Airline_Other, 
Airline_SpiceJet, Airline_Vistara, Source_Chennai, Source_Kolkata, Source_Mumbai, 
Destination_Cochin, Destination_Delhi, Destination_Hyderabad, Destination_Kolkata]
```

**Note**: `Source_Delhi` is deliberately dropped after VIF multicollinearity analysis to prevent perfect linear dependence (since one source city is always either present or absent across all four binary columns).

---

## Notes

- The notebook uses `drop_first=True` in one-hot encoding to prevent multicollinearity
- `Source_Delhi` is dropped after VIF analysis to further reduce redundancy
- Duration column parsing handles edge cases (missing hours or minutes)
- Random state is set to 42 for reproducible splits
- No hyperparameter tuning is performed; default Random Forest settings are used
- The feature engineering process is manually coded and must be replicated in `app.py` for serving predictions

