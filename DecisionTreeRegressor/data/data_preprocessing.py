# import pandas as pd
# import os

# # # Get the directory where this script is located
# # script_dir = os.path.dirname(os.path.abspath(__file__))
# # csv_path = os.path.join(script_dir, 'insurance.csv')

# df = pd.read_csv("insurance.csv")   # <-- use df, not train_df
# # Handle categorical variables
# data_prc = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
# data_prc.to_csv('preprocessed_insurance.csv', index=False)
# # Separate features and target variable
# #X = data_prc.drop('charges', axis=1).values
# #y = data_prc['charges'].values


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load raw data
df = pd.read_csv('insurance.csv')

# Encode categorical columns
label_encoders = {}
for col in ['sex', 'smoker', 'region']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # optional, if you need to decode later

# Save the processed CSV (without index, include header)
df.to_csv('preprocessed_insurance.csv', index=False)
print("Processed data saved to insurance_processed.csv")
print("Column order:", list(df.columns))