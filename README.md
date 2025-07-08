# 🏦 Bank Complaint Classifier (Multilingual)

This web application, developed using Streamlit, is designed to categorize customer complaints pertaining to **banking and finance** into distinct classifications through the utilization of a trained machine learning model.

🔍 It accommodates **multilingual input** by employing automatic translation and effectively filters out invalid or non-financial complaints.

---

## 📌 Features

- ✅ Categorizes financial complaints into established classifications
- 🌍 Accepts input in various languages (automatically translated to English)
- 🛡️ Identifies gibberish, irrelevant, or non-financial content
- 📊 Developed with Scikit-learn and Naive Bayes
- ⚡ Rapid user interface powered by Streamlit

## 🧠 Model Information

- The model has been trained utilizing `Multinomial Naive Bayes`
- The input has been vectorized with `CountVectorizer`
- Label encoding has been performed using `LabelEncoder`
- Data: A custom CSV file containing labeled bank complaints (not provided due to its size)
