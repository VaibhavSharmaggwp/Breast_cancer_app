import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Step 1: Create output folder if it doesn't exist
output_dir = "Assignment_7"
os.makedirs(output_dir, exist_ok=True)

# Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Convert to DataFrame for EDA
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

# Save feature names for use in app.py
with open(os.path.join(output_dir, "feature_names.pkl"), "wb") as f:
    pickle.dump(feature_names, f)

# EDA: Feature Importance Plot
feature_importance = model.feature_importances_
feature_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
feature_df = feature_df.sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(feature_df["Feature"][:10], feature_df["Importance"][:10])
plt.xlabel("Importance")
plt.title("Top 10 Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.close()

# EDA: Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# EDA: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Features")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()

# EDA: Pair Plot for Top 4 Features
top_features = feature_df["Feature"][:4].values
sns.pairplot(df, vars=top_features, hue="target", palette={0: "red", 1: "blue"})
plt.suptitle("Pair Plot of Top 4 Important Features", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pair_plot.png"))
plt.close()

# EDA: Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df["target"], palette={0: "red", 1: "blue"})
plt.xticks([0, 1], ["Malignant", "Benign"])
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "class_distribution.png"))
plt.close()
