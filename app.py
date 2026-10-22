
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from pathlib import Path

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)


@st.cache_resource
def train_model():
    # Load the SMS dataset stored in the GitHub repository
    data_path = Path(__file__).parent / "spam.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            "spam.csv is missing from the GitHub repository."
        )

    data = pd.read_csv(data_path, encoding="latin-1")

    # The original dataset uses v1 for labels and v2 for messages
    data = data[["v1", "v2"]].dropna()
    data.columns = ["label", "message"]

    data["label"] = data["label"].str.strip().str.lower()
    data = data[data["label"].isin(["ham", "spam"])]

    if data.empty:
        raise ValueError("No valid SMS messages found in spam.csv.")

    # Train a fresh vectorizer and classifier together
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2)
            )
        ),
        ("classifier", MultinomialNB())
    ])

    pipeline.fit(data["message"], data["label"])
    return pipeline


st.title("📩 SMS Spam Classifier")
st.write("Enter a text message to check whether it is spam.")

try:
    model = train_model()

    message = st.text_area(
        "Enter your SMS message",
        placeholder="Example: Congratulations! You have won a free prize."
    )

    if st.button("Predict", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            prediction = model.predict([message])[0]
            probabilities = model.predict_proba([message])[0]
            confidence = max(probabilities) * 100

            if prediction == "spam":
                st.error("🚨 Prediction: SPAM")
            else:
                st.success("✅ Prediction: NOT SPAM")

            st.write(f"Model confidence: {confidence:.2f}%")

except Exception as error:
    st.error(f"Unable to load or train the model: {error}")
