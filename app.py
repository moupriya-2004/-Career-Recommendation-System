import streamlit as st
import numpy as np
import pickle

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
label_encoder_work = pickle.load(open("label_encoder_work.pkl", "rb"))
label_encoder_career = pickle.load(open("label_encoder_career.pkl", "rb"))

st.set_page_config(page_title="Career Recommender", page_icon="🎯")
st.title("🎯 AI Career Recommendation System")

skills = st.text_input("Enter your skills (e.g., Coding, Testing):")
interests = st.text_input("Enter your interests (e.g., AI, Management):")
academic_score = st.slider("Academic Score (out of 10)", 0.0, 10.0, 7.5)
preferred_work = st.selectbox("Preferred Work Type", list(label_encoder_work.classes_))

if st.button("Recommend Career"):
    try:
        work_encoded = label_encoder_work.transform([preferred_work])[0]
        input_features = np.array([[work_encoded, academic_score]])
        prediction_encoded = model.predict(input_features)[0]
        recommended_career = label_encoder_career.inverse_transform([prediction_encoded])[0]
        st.success(f"✅ Recommended Career Path: **{recommended_career}**")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
