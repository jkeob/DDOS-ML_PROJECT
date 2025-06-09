# === Random Forest Results ===
#               precision    recall  f1-score   support
#
#            0       0.31      0.56      0.40      1459
#            1       0.91      0.79      0.85      8520
#
#     accuracy                           0.76      9979
#    macro avg       0.61      0.68      0.62      9979
# weighted avg       0.83      0.76      0.78      9979


#####################################################################################
import pandas as pd
import numpy as np
import joblib  # for saving the model
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


#Load the data set and use only the first 5000 rows. Had a low memory issues so i set it to false.
df = pd.read_csv(r"C:\Users\jpo2k\OneDrive\Desktop\Research Project\01-12\DrDoS_DNS.csv",nrows=10000, low_memory=False)
df3 = pd.read_csv(r"C:\Users\jpo2k\OneDrive\Desktop\Research Project\01-12\DrDoS_UDP.csv",nrows=10000, low_memory=False)
df4 = pd.read_csv(r"C:\Users\jpo2k\OneDrive\Desktop\Research Project\01-12\Syn.csv",nrows=10000, low_memory=False)
df5 = pd.read_csv(r"C:\Users\jpo2k\OneDrive\Desktop\Research Project\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",nrows=10000, low_memory=False)
df2 = pd.read_csv(r"C:\Users\jpo2k\OneDrive\Desktop\Research Project\data.csv",nrows=10000, low_memory=False)

#make sure the csv file loaded
print("CSV loaded successfully.")
print(df.head())  # just to make sure data looks right
print(df2.head())  # just to make sure data looks right
print(df3.head())  # just to make sure data looks right
print(df4.head())  # just to make sure data looks right
print(df5.head())  # just to make sure data looks right


# keep only relevant rows (normal + 1 attack type)
df = df.dropna()
df2 = df2.dropna()
df3 = df3.dropna()
df4 = df4.dropna()
df5 = df5.dropna()
df.columns = df.columns.str.strip()  # strip the whitespace from all column names
df2.columns = df2.columns.str.strip()  # strip the whitespace from all column names
df3.columns = df3.columns.str.strip()  # strip the whitespace from all column names
df4.columns = df4.columns.str.strip()  # strip the whitespace from all column names
df5.columns = df5.columns.str.strip()  # strip the whitespace from all column names


# print("Exact column names in df:")
# for col in df.columns:
#     print(f"'{col}'")
# print("Exact column names in df2:")
# for col in df2.columns:
#     print(f"'{col}'")
# print("Exact column names in df3:")
# for col in df3.columns:
#     print(f"'{col}'")
# print("Exact column names in df4:")
# for col in df4.columns:
#     print(f"'{col}'")
# print("Exact column names in df5:")
# for col in df5.columns:
#     print(f"'{col}'")

# Filter and relabel both datasets
df = df[df['Label'].isin(['BENIGN', 'DrDoS_DNS'])]
df['Label'] = df['Label'].map({'BENIGN': 0, 'DrDoS_DNS': 1})

df2 = df2[df2['Label'].isin(['Benign', 'Malicious'])]
df2['Label'] = df2['Label'].map({'Benign': 0, 'Malicious': 1})

df3 = df3[df3['Label'].isin(['BENIGN', 'DrDoS_UDP'])]
df3['Label'] = df3['Label'].map({'BENIGN': 0, 'DrDoS_UDP': 1})

df4 = df4[df4['Label'].isin(['BENIGN', 'Syn'])]
df4['Label'] = df4['Label'].map({'BENIGN': 0, 'Syn': 1})

df5 = df5[df5['Label'].isin(['Benign', 'Bot'])]
df5['Label'] = df5['Label'].map({'Benign': 0, 'Bot': 1})


df2.rename(columns={
    'Total Fwd Packet': 'Total Fwd Packets',
    'Total Bwd packets': 'Total Backward Packets',
    'Total Length of Fwd Packet': 'Total Length of Fwd Packets',
    'Total Length of Bwd Packet': 'Total Length of Bwd Packets',
}, inplace=True)

df5.rename(columns={
    'Tot Fwd Pkts': 'Total Fwd Packets',
    'Tot Bwd Pkts': 'Total Backward Packets',
    'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
    'TotLen Bwd Pkts': 'Total Length of Bwd Packets'
}, inplace=True)



# Final feature list
features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets']

# Combine only the training datasets (excluding df5)
train_df = pd.concat([df, df2, df3, df4], ignore_index=True)
X_train = train_df[features]
y_train = train_df['Label']

# Use df5 exclusively as the test set
X_test = df5[features]
y_test = df5['Label']
#commit new

# Debug: check how many rows are valid
# print("\n=== Feature Check ===")
# print("Shape of X before dropna:", X.shape)
# print("NaNs per feature:\n", X.isnull().sum())
# print("Total rows with any missing feature values:", X.isnull().any(axis=1).sum())
# print("Rows remaining after dropna:", X.dropna().shape[0])


# train the Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


print("=== Random Forest Results ===")
print(classification_report(y_test, y_pred))

# Save the trained model to a .pkl file so it can be used later for live detection. This is the program that will be going into the
#Rasberry Pi
joblib.dump(rf, 'ddos_rf_model.pkl')
print("Model saved to ddos_rf_model.pkl")
