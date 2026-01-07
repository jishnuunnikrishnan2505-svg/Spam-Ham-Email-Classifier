import streamlit as st
import joblib

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Detection")
st.write("Enter any email text to check if it is **Spam** or **Ham**.")

# Load pipeline model
@st.cache_resource
def load_model():
    return joblib.load("Naive Bayes spam_classifier_model.pkl")

model = load_model()

# User input
email_text = st.text_area(
    "✉️ Paste email content here:",
    height=180,
    placeholder="Congratulations! You won a prize..."
)

if st.button("🔍 Check Email"):
    if email_text.strip() == "":
        st.warning("Please enter email text.")
    else:
        # ✅ FIX: pass text as LIST (2D expected by pipeline)
        prediction = model.predict([email_text])[0]

        if prediction == "spam":
            st.error("🚨 This email is **SPAM**")
        else:
            st.success("✅ This email is **NOT SPAM (HAM)**")
