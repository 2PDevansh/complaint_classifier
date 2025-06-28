import streamlit as st
import pickle


def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_model()


st.set_page_config(page_title="Complaint Classifier", layout="centered")


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
        background-color: #e6f2ff;
        color: #1a1a1a;
    }
    h1 {
        color: #004080;
        text-align: center;
        font-size: 42px;
        margin-bottom: 30px;
    }
    .stTextInput>div>div>input, .stTextArea>div>textarea {
        background-color: #ffffff;
        border: 1px solid #99c2ff;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #004080;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #0066cc;
    }
    .stMarkdown h4 {
        color: #004080;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### 📌 Instructions")
    st.write("""
    - Paste your complaint in the box.
    - Click **Classify** to see the predicted category.
    - Make sure models are loaded.
    """)


st.markdown("<h1>🔍 Complaint Category Classifier</h1>", unsafe_allow_html=True)


complaint_text = st.text_area("📝 Enter your complaint text below:", height=180)


if st.button("Classify"):
    if complaint_text.strip() == "":
        st.warning("Please enter a complaint first.")
    else:
        X = vectorizer.transform([complaint_text])
        pred = model.predict(X)
        label = label_encoder.inverse_transform([pred[0]])[0]
        st.success(f"✅ **Predicted Category:** {label}")
