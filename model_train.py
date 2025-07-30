import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("career_dataset.csv")

# Encode categorical features
le_work = LabelEncoder()
le_career = LabelEncoder()

df["Preferred_Work_Encoded"] = le_work.fit_transform(df["Preferred_Work"])
df["Career_Encoded"] = le_career.fit_transform(df["Career"])

X = df[["Preferred_Work_Encoded", "Academic_Score"]]
y = df["Career_Encoded"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le_work, open("label_encoder_work.pkl", "wb"))
pickle.dump(le_career, open("label_encoder_career.pkl", "wb"))

print("✅ Model trained and saved successfully!")
