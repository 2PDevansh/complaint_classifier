import streamlit as st
import pickle
import re
from googletrans import Translator


def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_model()
translator = Translator()


st.set_page_config(page_title="Bank Complaint Classifier", page_icon="🔎", layout="wide")

# Styling
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #e6f2ff;
        color: #1a1a1a;
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
    </style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### 📌 Instructions")
    st.write("""
    - Enter your complaint in **any language**.
    - The app auto-translates and classifies it.
    - Make sure it's a real issue, not polite text.
    """)


st.markdown("<h3 style='color: navy;'>🏦 Bank Complaint Classifier (Multilingual)</h3>", unsafe_allow_html=True)


complaint_text = st.text_area("📝 Enter your complaint text below:", height=180)

# Strong validation function
def is_invalid_input(text):
    text = text.strip()
    if len(text) < 5:
        return True, "❗ Complaint is too short."

    if re.fullmatch(r'[^a-zA-Z0-9\s]+', text) or re.fullmatch(r'(.)\\1{3,}', text):
        return True, "❗ Complaint looks like gibberish."

    try:
        detected_lang = translator.detect(text).lang
    except Exception:
        return True, "❗ Unable to detect language."

    # If English, check for complaint-related keywords
    if detected_lang == 'en':
        complaint_keywords = [
            'not', 'unable', 'issue', 'problem', 'complaint',
            'error', 'fail', 'decline', 'blocked', 'refund', 'charged', 'wrong', 'missing', 'late','money', 'service', 'support', 
        ]
        if not any(word in text.lower() for word in complaint_keywords):
            return True, "❗ This doesn't appear to describe a complaint. Please describe the issue clearly."

    return False, ""

# Classify button
if st.button("Classify"):
    if complaint_text.strip() == "":
        st.warning("Please enter a complaint first.")
    else:
        # Validate input
        is_invalid, reason = is_invalid_input(complaint_text)
        if is_invalid:
            st.warning(reason)
        else:
            try:
                # Translate to English if needed
                translated = translator.translate(complaint_text, dest='en')
                translated_text = translated.text.strip()
                detected_lang = translated.src

                st.markdown(f"🌐 Detected Language: `{detected_lang.upper()}`")
                if detected_lang != 'en':
                    st.markdown("🔁 **Translated Complaint:**")
                    st.info(translated_text)

                # Predict
                X = vectorizer.transform([translated_text])
                pred = model.predict(X)
                label = label_encoder.inverse_transform([pred[0]])[0]
                st.success(f"✅ **Predicted Category:** {label}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
