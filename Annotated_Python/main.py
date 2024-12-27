# Import necessary libraries
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the ARFF file and convert to DataFrame
file_path = "/Users/thisarurathnayake/Downloads/Adults data set (for labs 2-4)-20241227/dataset_adult.arff"
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# Decode binary columns to strings (necessary for proper handling of categorical data)
for col in df.select_dtypes([object]).columns:
    df[col] = df[col].str.decode('utf-8')

# Display a preview of the dataset
print("Dataset sample:")
print(df.head())

# Step 2: Preprocess the dataset
# Encode categorical variables using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate features (X) and target (y)
X = df_encoded.drop('income_>50K', axis=1)
y = df_encoded['income_>50K']

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Evaluate model performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Save the preprocessed dataset for reproducibility
import os

output_dir = '/mnt/data'  # Change this to your desired directory
os.makedirs(output_dir, exist_ok=True)  # This will create the directory if it doesn't exist
output_file = os.path.join(output_dir, 'output.csv')  # Adjust this for your file name
df_encoded.to_csv(output_file, index=False)

output_file = '/Users/thisarurathnayake/Documents/output.csv'  # Adjust to your desired path
df_encoded.to_csv(output_file, index=False)



output_file = '/mnt/data/adult_dataset_preprocessed.csv'
df_encoded.to_csv(output_file, index=False)
print(f"Preprocessed dataset saved to {output_file}")

# Optional: Display feature importance
feature_importances = clf.feature_importances_
important_features = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
important_features = important_features.sort_values(by='Importance', ascending=False)
print("Top features contributing to predictions:")
print(important_features.head())


