import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from db import load_data, append_row

LABEL_COLS = ["Depression", "Anxiety", "Insomnia", "Social Anxiety", "Psychotic Symptoms"]

YES_MSG = {
    "Depression": "🧠 You may be experiencing signs of low mood or sadness.",
    "Anxiety": "😰 You may be showing signs of restlessness or tension.",
    "Insomnia": "🌙 Your sleep pattern suggests possible difficulty falling or staying asleep.",
    "Social Anxiety": "🙈 You may have signs of social withdrawal or discomfort in interactions.",
    "Psychotic Symptoms": "🔊 You might be experiencing perceptual disturbances like hearing whispers."
}

NO_MSG = {
    "Depression": "✅ No major signs of prolonged sadness.",
    "Anxiety": "✅ You appear calm and emotionally stable.",
    "Insomnia": "✅ Your sleep pattern looks normal.",
    "Social Anxiety": "✅ No signs of social discomfort detected.",
    "Psychotic Symptoms": "✅ No perceptual disturbance patterns found."
}

def preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    df.rename(columns={
        "Do you often feel low or sad for several days in a row?": "Depression",
        "Do you feel anxious, tense, or restless without any clear reason?": "Anxiety",
        "Do you find it difficult to fall asleep or stay asleep?": "Insomnia",
        "Do you avoid talking or meeting people, even if you want to?": "Social Anxiety",
        "Have you ever heard someone call your name or whisper when no one was around?": "Psychotic Symptoms",
        "How confident or positive do you feel about yourself (self-esteem)?": "Self Esteem"
    }, inplace=True)

    yes_no_cols = [
        *LABEL_COLS,
        "Have you ever been diagnosed with a mental health disorder by a professional?",
        "Do you have a family history of mental illness (parents, siblings, etc.)?",
        "Are you currently taking any medication for mental wellness, stress, or anxiety?",
        "Do you engage in any physical activity (e.g., walking, gym, yoga) daily?",
        "Do you usually use screens after 10 PM at night?"
    ]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "Not sure": 0, "Prefer not to say": 0})

    for col in LABEL_COLS:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna("0").astype(str).str.strip().replace({"": "0"}).astype(int)


    cat_cols = ["Gender", "Educational Level", "Employment Status", "How would you rate your caffeine intake?"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    numeric_cols = [
        "Age", "How many hours do you sleep per day (on average)?",
        "How many hours do you use mobile, computer, or TV daily?",
        "If you work or study, how many hours do you do so daily?",
        "How many meals do you eat per day (on average)?",
        "Self Esteem"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    features = [
        "Age", "Gender", "Educational Level", "Employment Status",
        "How many hours do you sleep per day (on average)?",
        "How many hours do you use mobile, computer, or TV daily?",
        "If you work or study, how many hours do you do so daily?",
        "Do you engage in any physical activity (e.g., walking, gym, yoga) daily?",
        "Do you usually use screens after 10 PM at night?",
        "How would you rate your caffeine intake?",
        "How many meals do you eat per day (on average)?",
        "Self Esteem"
    ]

    df[features] = df[features].fillna(0)

    X = df[features]
    y = df[LABEL_COLS]

    trainable_labels = [label for label in LABEL_COLS if y[label].nunique() > 1]
    if len(trainable_labels) < len(LABEL_COLS):
        st.warning("⚠️ Some labels had no variety and were excluded from training: " +
                   ", ".join([label for label in LABEL_COLS if label not in trainable_labels]))

    if not trainable_labels:
        st.error("❌ None of the labels had enough variety to train.")
        return None, None, features, df, []  # ✅ Make sure this is 5 values

    y = y[trainable_labels]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        model = MultiOutputClassifier(
            LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000)
        )
        model.fit(X_scaled, y)
    except ValueError as e:
        st.warning(f"Model training failed: {e}")
        return None, None, features, df, []

    return model, scaler, features, df, trainable_labels

def main():
    st.set_page_config(page_title="Mental Health Predictor", page_icon="🧠")
    st.title("🧠 Lifestyle & Mental Health Predictor")
    st.caption("Get feedback based on your daily habits. This does not replace a clinical diagnosis.")

    raw_df = load_data()
    model, scaler, features, df, trainable_labels = preprocess(raw_df)
    model_ready = model is not None

    if model_ready:
        st.success("✅ Model trained successfully.")
        st.subheader("📊 Current Trends from Dataset")
        col1, col2 = st.columns(2)
        for i, label in enumerate(trainable_labels):
            pct = 100 * df[label].mean()
            (col1 if i % 2 == 0 else col2).metric(label, f"{pct:.1f}% likely")
    else:
        st.warning("⚠️ Model could not be trained yet. Waiting for more responses.")

    st.divider()
    st.subheader("🧾 Check Your Own Lifestyle Health")

    with st.form("user_form", clear_on_submit=True):
        age = st.slider("Age", 12, 65, 22)
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        edu = st.selectbox("Education", ["High School", "Diploma", "Graduate", "Postgraduate", "Other"])
        emp = st.selectbox("Employment", ["Student", "Employed", "Unemployed"])
        sleep = st.slider("Sleep (hrs)", 3.0, 12.0, 7.0)
        screen = st.slider("Screen time (hrs)", 1.0, 16.0, 6.0)
        work = st.slider("Study/Work (hrs)", 0.0, 12.0, 6.0)
        activity = st.radio("Daily exercise?", ["Yes", "No"], horizontal=True)
        late_scr = st.radio("Screen after 10 PM?", ["Yes", "No"], horizontal=True)
        caffeine = st.selectbox("Caffeine intake", ["Low", "Moderate", "High"])
        meals = st.slider("Meals/day", 1, 5, 3)
        esteem = st.slider("Self esteem (1 10)", 1, 10, 6)
        consent = st.checkbox("✅ Allow my anonymous data to be used for research")
        submit = st.form_submit_button("💡 Analyze Me")

    if submit:
        row = {
            "Age": age,
            "Gender": ["Male", "Female", "Other", "Prefer not to say"].index(gender),
            "Educational Level": ["High School", "Diploma", "Graduate", "Postgraduate", "Other"].index(edu),
            "Employment Status": ["Student", "Employed", "Unemployed"].index(emp),
            "How many hours do you sleep per day (on average)?": sleep,
            "How many hours do you use mobile, computer, or TV daily?": screen,
            "If you work or study, how many hours do you do so daily?": work,
            "Do you engage in any physical activity (e.g., walking, gym, yoga) daily?": 1 if activity == "Yes" else 0,
            "Do you usually use screens after 10 PM at night?": 1 if late_scr == "Yes" else 0,
            "How would you rate your caffeine intake?": ["Low", "Moderate", "High"].index(caffeine),
            "How many meals do you eat per day (on average)?": meals,
            "Self Esteem": esteem
        }

        if model_ready:
            X_user = pd.DataFrame([row])[features]
            preds = model.predict(scaler.transform(X_user))[0]
            st.subheader("🧠 Your Personalized Feedback")
            for i, label in enumerate(trainable_labels):
                st.write(YES_MSG[label] if preds[i] else NO_MSG[label])
        else:
            st.info("Thanks! Your data was recorded. We need more responses for prediction.")

        if consent:
            record = row.copy()
            record["timestamp"] = datetime.datetime.now().isoformat()
            record["source"] = "human"
            if model_ready:
                for i, label in enumerate(trainable_labels):
                    record[label] = int(preds[i])
            append_row(record)
            st.success("📝 Your response has been saved.")

        st.subheader("📋 Your Submitted Details")
        st.json(row)

if __name__ == "__main__":
    main()