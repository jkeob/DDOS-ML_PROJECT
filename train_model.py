

# === Random Forest Results ===
#               precision    recall  f1-score   support

#            0       0.84      1.00      0.91     99919
#            1       1.00      0.80      0.89     91593

#     accuracy                           0.90    191512
#    macro avg       0.92      0.90      0.90    191512
# weighted avg       0.92      0.90      0.90    191512
#####################################################################################
import json
import os
import pandas as pd
import numpy as np
import joblib  # for saving the model
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score


#Load the data set and use only the first 5000 rows. Had a low memory issues so i set it to false.
df = pd.read_csv(r"C:\Users\jpo2k\PycharmProjects\RESEARCHPROJECT\datasets\DrDoS_DNS.csv",nrows=100000, low_memory=False)
df3 = pd.read_csv(r"C:\Users\jpo2k\PycharmProjects\RESEARCHPROJECT\datasets\DrDoS_UDP.csv",nrows=100000, low_memory=False)
df4 = pd.read_csv(r"C:\Users\jpo2k\PycharmProjects\RESEARCHPROJECT\datasets\Syn.csv",nrows=100000, low_memory=False)
df5 = pd.read_csv(r"C:\Users\jpo2k\PycharmProjects\RESEARCHPROJECT\datasets\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",nrows= 10000, low_memory=False)
df6 = pd.read_csv(r"C:\Users\jpo2k\PycharmProjects\RESEARCHPROJECT\datasets\Monday-WorkingHours.pcap_ISCX-FULLBENIGN.csv",nrows= 100000, low_memory=False)
df7 = pd.read_csv(r"C:\Users\jpo2k\PycharmProjects\RESEARCHPROJECT\datasets\Wednesday-workingHours.pcap_ISCX-SLOWLORIS.csv",nrows= 20000, low_memory=False)
df8 = pd.read_csv(r"C:\Users\jpo2k\PycharmProjects\RESEARCHPROJECT\datasets\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",nrows= 20000, low_memory=False)
df9 = pd.read_csv(r"C:\Users\jpo2k\PycharmProjects\RESEARCHPROJECT\datasets\test_formatted_pcap_to_csv.csv", low_memory=False)
df2 = pd.read_csv(r"C:\Users\jpo2k\PycharmProjects\RESEARCHPROJECT\datasets\data.csv",nrows=100000, low_memory=False)

#make sure the csv file loaded
print("CSV loaded successfully.")

# keep only relevant rows (normal + 1 attack type)
df = df.dropna()
df2 = df2.dropna()
df3 = df3.dropna()
df4 = df4.dropna()
df5 = df5.dropna()
df6 = df6.dropna()
df7 = df7.dropna()
df8 = df8.dropna()
df9 = df9.dropna()



df.columns = df.columns.str.strip()  # strip the whitespace from all column names
df2.columns = df2.columns.str.strip()  # strip the whitespace from all column names
df3.columns = df3.columns.str.strip()  # strip the whitespace from all column names
df4.columns = df4.columns.str.strip()  # strip the whitespace from all column names
df5.columns = df5.columns.str.strip()  # strip the whitespace from all column names
df6.columns = df6.columns.str.strip()  # strip the whitespace from all column names
df7.columns = df7.columns.str.strip()  # strip the whitespace from all column names
df8.columns = df8.columns.str.strip()  # strip the whitespace from all column names
df9.columns = df9.columns.str.strip()  # strip the whitespace from all column names


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

df6 = df6[df6['Label'].isin(['BENIGN', 'DrDoS_DNS'])]
df6['Label'] = df6['Label'].map({'BENIGN': 0, 'DrDoS_DNS': 1})

df7['Label'] = df7['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)

#
# df8['Label'] = df8['Label'].astype(str).str.strip().str.upper()
# print(df8['Label'].value_counts())
df8 = df8[df8['Label'].isin(['BENIGN', 'DDoS'])]
df8['Label'] = df8['Label'].map({'BENIGN': 0, 'DDoS': 1})

df9 = df9[df9['Label'].isin(['BENIGN', 'DDoS'])]
df9['Label'] = df9['Label'].map({'BENIGN': 0, 'DDoS': 1})




df2.rename(columns={
    'Dst Port': 'Destination Port',
    'Total Fwd Packet': 'Total Fwd Packets',
    'Total Bwd packets': 'Total Backward Packets',
    'Total Length of Fwd Packet': 'Total Length of Fwd Packets',
    'Total Length of Bwd Packet': 'Total Length of Bwd Packets',
    'FWD Init Win Bytes': 'Init_Win_bytes_forward',
    'Bwd Init Win Bytes': 'Init_Win_bytes_backward',
}, inplace=True)


df5.rename(columns={
    'Dst Port': 'Destination Port',
    'Tot Fwd Pkts': 'Total Fwd Packets',
    'Tot Bwd Pkts': 'Total Backward Packets',
    'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
    'TotLen Bwd Pkts': 'Total Length of Bwd Packets',
    'Flow Byts/s': 'Flow Bytes/s',
    'Flow Pkts/s': 'Flow Packets/s',
    'SYN Flag Cnt': 'SYN Flag Count',
    'ACK Flag Cnt': 'ACK Flag Count',
    'FIN Flag Cnt': 'FIN Flag Count',
    'PSH Flag Cnt': 'PSH Flag Count',
    'URG Flag Cnt': 'URG Flag Count',
    'Fwd Header Len': 'Fwd Header Length',
    'Bwd Header Len': 'Bwd Header Length',
    'Init Fwd Win Byts': 'Init_Win_bytes_forward',
    'Init Bwd Win Byts': 'Init_Win_bytes_backward',
    'Pkt Size Avg': 'Average Packet Size'
}, inplace=True)




#feature list
features = [
    'Destination Port',
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Flow Bytes/s',
    'Flow Packets/s',
    'SYN Flag Count',
    'ACK Flag Count',
    'FIN Flag Count',
    'PSH Flag Count',
    'URG Flag Count',
    'Fwd Header Length',
    'Bwd Header Length',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'Average Packet Size'
]

# Split df6 into train/test portions

# Add df6_train to training set
train_df = pd.concat([df,df2,df3,df5,df7,df8], ignore_index=True)
X_train = train_df[features]
y_train = train_df['Label']

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]



# Prepare test set (right after df9 is processed)
df_test = pd.concat([df4, df6], ignore_index=True)  # or another benign-heavy set
df_test = df_test[df_test['Label'].isin([0, 1])]    # ensure it's clean


X_test = df_test[features]
y_test = df_test['Label']

X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]

# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import classification_report
#
# # 10-fold cross-validation
# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#
# print("=== 10-Fold Classification Reports ===")
# fold = 1
# for train_index, val_index in cv.split(X_train, y_train):
#     X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
#     y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
#
#     rf = RandomForestClassifier(
#         n_estimators=100,
#         max_depth=None,
#         min_samples_split=3,
#         min_samples_leaf=2,
#         class_weight={0: 1.0, 1: 2.25},
#         random_state=42,
#         n_jobs=-1
#     )
#     rf.fit(X_tr, y_tr)
#     y_pred = rf.predict(X_val)
#
#     print(f"\n--- Fold {fold} ---")
#     print(classification_report(y_val, y_pred, digits=4))
#     fold += 1
#
# # Train on full X_train
# rf.fit(X_train, y_train)
#
# # Evaluate on completely separate data (e.g., df9)
# y_pred = rf.predict(X_test)
# print("=== Final Test Set Results ===")
# print(classification_report(y_test, y_pred, digits=4))






# Debug: check how many rows are valid
# print("\n=== Feature Check ===")
# print("Shape of X before dropna:", X.shape)
# print("NaNs per feature:\n", X.isnull().sum())
# print("Total rows with any missing feature values:", X.isnull().any(axis=1).sum())
# print("Rows remaining after dropna:", X.dropna().shape[0])


# train the Random Forest Classifier

n_runs = 1
all_reports = []

for i in range(n_runs):

    xgb = XGBClassifier(
        n_estimators=250,
        learning_rate=0.1,
        max_depth=8,
        scale_pos_weight=2.25,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)

    y_probs = xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.05).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    all_reports.append((precision, recall, f1))

# Convert to numpy array for easy averaging
all_reports = np.array(all_reports)
avg_precision, avg_recall, avg_f1 = all_reports.mean(axis=0)


print("=== Random Forest Results ===")
print(classification_report(y_test, y_pred))
#
# Save the trained model to a .pkl file so it caAn be used later for live detection. This is the program that will be going into the
# Rasberry Pi
model_path = os.path.join('models', 'ddos_rf_model.pkl')
joblib.dump(xgb, model_path)
print(f"✅ Model saved to {model_path}")

# Save features
feature_list = [
    'Destination Port',
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Flow Bytes/s',
    'Flow Packets/s',
    'SYN Flag Count',
    'ACK Flag Count',
    'FIN Flag Count',
    'PSH Flag Count',
    'URG Flag Count',
    'Fwd Header Length',
    'Bwd Header Length',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'Average Packet Size'
]


feature_path = os.path.join('models', 'ddos_rf_features.json')
with open(feature_path, 'w') as f:
    json.dump(feature_list, f)
print(f"✅ Feature list saved to {feature_path}")
