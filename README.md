# 🏦 Bank Complaint Classifier (Multilingual)

This is a Streamlit-based web app that classifies customer complaints related to **banking and finance** into specific categories using a trained machine learning model.

🔍 It supports **multilingual input** via automatic translation and filters out invalid or non-financial complaints.

---

## 📌 Features

- ✅ Classifies financial complaints into predefined categories
- 🌍 Supports input in multiple languages (auto-translated to English)
- 🛡️ Detects gibberish, irrelevant, or non-financial text
- 📊 Built with Scikit-learn + Naive Bayes
- ⚡ Fast UI with Streamlit

## 🧠 Model Information

- The model has been trained utilizing `Multinomial Naive Bayes`
- The input has been vectorized with `CountVectorizer`
- Label encoding has been performed using `LabelEncoder`
- Data: A custom CSV file containing labeled bank complaints (not provided due to its size)
