import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 1: Load the dataset
try:
    df = pd.read_csv("heart_cleveland_upload.csv")
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: CSV file not found! Please check the file path.")
    exit()

# Step 2: Debugging - Print column names
print("Columns in dataset:", df.columns)

# Step 3: Set Target Column
TARGET_COLUMN = "condition"  # We now use "condition" as the target column
if TARGET_COLUMN not in df.columns:
    print("❌ Error: Target column not found! Please check your dataset.")
    exit()

print(f"✅ Target column found: {TARGET_COLUMN}")

# Step 4: Split Features & Target
X = df.drop(columns=[TARGET_COLUMN])  # Drop target column
y = df[TARGET_COLUMN]  # Target variable

# Step 5: Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# Step 7: Save Model
with open("heart_disease_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
print("✅ Model saved as 'heart_disease_model.pkl'")