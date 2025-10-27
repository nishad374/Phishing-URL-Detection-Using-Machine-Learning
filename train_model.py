import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create pickle directory if it doesn't exist
if not os.path.exists('pickle'):
    os.makedirs('pickle')

# Load dataset
data = pd.read_csv("phishing.csv")

# Rename target column 'class' to 'Label' to match the code
data.rename(columns={'class': 'Label'}, inplace=True)

# Check the structure of the dataset
print("Dataset shape:", data.shape)
print("Columns:", data.columns.tolist())
print("First few rows:")
print(data.head())

# Check for missing values
print("Missing values:")
print(data.isnull().sum())

# Check the distribution of the target variable
print("Target value distribution:")
print(data['Label'].value_counts())

# Features (X) and Labels (y)
X = data.drop('Label', axis=1)
y = data['Label']

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("Class distribution after SMOTE:")
print(pd.Series(y_res).value_counts())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model - try both Gradient Boosting and Random Forest
models = {
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
    
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{name.lower()}_confusion_matrix.png')
    plt.close()

# Save the best model along with the scaler
with open("pickle/model.pkl", "wb") as file:
    pickle.dump({'model': best_model, 'scaler': scaler}, file)

print(f"\n✅ Best model saved as pickle/model.pkl with accuracy: {best_accuracy:.4f}")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
