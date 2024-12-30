# Human Activity Recognition using Machine Learning

This repository demonstrates how to perform Human Activity Recognition (HAR) using a time-series dataset of sensor data. The goal is to classify human activities such as walking, running, or sitting based on x, y, and z-axis accelerometer data.

## Dataset
The dataset used in this project is stored in the file `time_series_data_human_activities.csv`.

### Example Columns:
- `timestamp`: The timestamp of the sensor reading.
- `x-axis`, `y-axis`, `z-axis`: Accelerometer readings along the three axes.
- `activity`: The activity label (e.g., walking, running, etc.).

## Workflow

### 1. Load the Dataset
Use pandas to load and inspect the dataset.

```python
import pandas as pd

# Load the dataset
file_path = 'time_series_data_human_activities.csv'
df = pd.read_csv(file_path)

# Display the first few rows
print(df.head())
```

### 2. Data Preprocessing
- Handle missing values.
- Encode the `activity` labels into numerical values.

```python
# Check for missing values
print(df.isnull().sum())

# Encode activity labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['activity'] = le.fit_transform(df['activity'])
```

### 3. Train-Test Split
Split the dataset into training and testing sets.

```python
from sklearn.model_selection import train_test_split

# Separate features and labels
X = df[['x-axis', 'y-axis', 'z-axis']]  # Sensor data
y = df['activity']                     # Labels

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4. Model Training
Train a Random Forest Classifier on the training data.

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)
```

### 5. Evaluate the Model
Check the model's accuracy on the test data.

```python
from sklearn.metrics import accuracy_score

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```

### 6. Save and Load the Model
Save the trained model for later use.

```python
import joblib

# Save the model
joblib.dump(clf, 'activity_recognition_model.pkl')

# Load the model
clf_loaded = joblib.load('activity_recognition_model.pkl')
```

## Repository Structure
```
.
├── time_series_data_human_activities.csv   # Dataset
├── activity_recognition_model.pkl          # Trained model (generated after training)
├── README.md                               # Documentation
└── har_model.py                            # Code for training and evaluation
```

## Requirements
Install the required Python libraries using pip:

```bash
pip install pandas scikit-learn matplotlib
```

## Usage
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Run the Python script:
   ```bash
   python har_model.py
   ```

3. Inspect the results and use the saved model for predictions.

## Future Improvements
- Add more advanced feature engineering.
- Implement deep learning models for improved accuracy.
- Visualize the data for better understanding of activity patterns.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
