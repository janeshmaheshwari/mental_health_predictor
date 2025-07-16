
# 🧠 Mental Health Predictor Web App

A user-friendly web application built with Python and Streamlit that helps identify the likelihood of five common mental health conditions based on lifestyle inputs. This project was developed as part of the IEEE EMBS Pune Internship.

---

## 🚀 Live Demo
Visit the deployed app here:  
🔗 [mentalhealthpredict.streamlit.app](https://mentalhealthpredict.streamlit.app)

---

## 🔍 What It Does

- Collects anonymous lifestyle information from users
- Trains a logistic regression model on existing response data
- Predicts the probability of the following conditions:
  - Depression
  - Anxiety
  - Insomnia
  - Social Anxiety
  - Psychotic Symptoms
- Displays real-time statistics on how many users are affected
- Allows users to contribute data to improve future predictions

---

## 📁 Project Structure

```
mental_health_predictor/
├── mental_health_app.py      # Main Streamlit application
├── db.py                     # SQLite DB handler for reading/writing
├── mental_health.db          # Stores all user responses (local DB)
├── responses.csv             # Backup CSV log of responses
├── requirements.txt          # Dependencies for deployment
└── .vscode/                  # Optional (VSCode task configs)
```

---

## 🛠 Technologies Used

- Python 3.10+
- Streamlit
- SQLite3
- Scikit-learn (Logistic Regression)
- Pandas, NumPy

---

## 🧪 How It Works

1. The user fills out a lifestyle questionnaire
2. The app preprocesses the input and passes it to a trained ML model
3. Predictions are shown with friendly feedback messages
4. If the user opts in, their anonymized data is stored for future model training
5. Stats from the growing dataset are displayed live on the home page

---

## 🏁 Getting Started Locally

### 1. Clone the repo
```bash
git clone https://github.com/janeshmaheshwari/mental_health_predictor.git
cd mental_health_predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run mental_health_app.py
```

---

## 🙋‍♂️ Author

**Janesh Maheshwari**  
IEEE EMBS Pune Internship 2025

---

## 🙏 Acknowledgements

- Guided by Dr. A Naresh Kumar
- Developed under IEEE EMBS Pune Internship Program
- Thanks to all contributors and test users

---

## 📌 Notes

- This app does not replace a professional mental health diagnosis.
- All data is stored locally and anonymized.
