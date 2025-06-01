import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


#Load the data set and use only the first 5000 rows. Had a low memory issues so i set it to false.
df = pd.read_csv(r"C:\Users\jpo2k\OneDrive\Desktop\Research Project\01-12\DrDoS_DNS.csv",nrows=5000, low_memory=False)

#make sure the csv file loaded
print("CSV loaded successfully.")
print(df.head())  # just to make sure data looks right

# keep only relevant rows (normal + 1 attack type)
df = df.dropna()
df.columns = df.columns.str.strip()  # strip the whitespace from all column names
print(df.columns)  # see all column names to confirm

df = df[df['Label'].isin(['BENIGN', 'DrDoS_DNS'])]
df['Label'] = df['Label'].map({'BENIGN': 0, 'DrDoS_DNS': 1})

# picking the relevant features — update these later
features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets']
X = df[features]
y = df['Label']

# split the data for training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train the Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("=== Random Forest Results ===")
print(classification_report(y_test, y_pred))

# Using Isolation Forest for the anomaly detection
iso = IsolationForest(contamination=0.1, random_state=42)
iso_preds = iso.fit_predict(X)
iso_preds = np.where(iso_preds == -1, 1, 0)  # convert -1 (anomaly) to 1

print("=== Isolation Forest Results ===")
print(confusion_matrix(y, iso_preds))

# Visualize data using a scatter plot.
plt.scatter(X['Flow Duration'], X['Total Fwd Packets'], c=y, cmap='coolwarm', s=10)
plt.title("Traffic Patterns - Flow Duration vs Forward Packets")
plt.xlabel("Flow Duration")
plt.ylabel("Tot Fwd Pkts")
plt.show()
