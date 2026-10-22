import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from pathlib import Path

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="wide"
)

# ---------------- MONGODB CONNECTION ----------------

@st.cache_resource
def get_collection():
    client = MongoClient(
        "mongodb://localhost:27017/",
        serverSelectionTimeoutMS=5000
    )
    client.admin.command("ping")

    db = client["sms_spam_db"]
    return db["sms_history"]


# ---------------- TRAIN ML MODEL ----------------

@st.cache_resource
def train_model():
    data_path = Path(__file__).parent / "spam.csv"

    if not data_path.exists():
        raise FileNotFoundError("spam.csv is missing.")

    data = pd.read_csv(data_path, encoding="latin-1")
    data = data[["v1", "v2"]].dropna()
    data.columns = ["label", "message"]

    data["label"] = data["label"].str.strip().str.lower()
    data = data[data["label"].isin(["ham", "spam"])]

    if data.empty:
        raise ValueError("No valid SMS messages found.")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )),
        ("classifier", MultinomialNB())
    ])

    pipeline.fit(data["message"], data["label"])
    return pipeline


# ---------------- INITIALIZE ----------------

st.title("📩 SMS Spam Classifier")
st.write("Detect spam messages using Machine Learning and manage records with MongoDB.")

try:
    model = train_model()
    collection = get_collection()

    # ---------------- DASHBOARD ----------------

    records = list(collection.find().sort("_id", -1))

    total = len(records)
    spam_count = sum(r["prediction"] == "spam" for r in records)
    ham_count = sum(r["prediction"] == "ham" for r in records)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total SMS", total)
    col2.metric("Spam SMS", spam_count)
    col3.metric("Not Spam SMS", ham_count)

    # ---------------- CREATE ----------------

    st.header("1. Predict SMS")

    with st.form("predict_form"):
        message = st.text_area(
            "Enter your SMS message",
            placeholder="Congratulations! You have won a free prize."
        )
        predict_button = st.form_submit_button(
            "Predict and Save",
            type="primary",
            use_container_width=True
        )

    if predict_button:
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            prediction = model.predict([message])[0]
            probabilities = model.predict_proba([message])[0]
            confidence = float(max(probabilities) * 100)

            collection.insert_one({
                "sms_text": message.strip(),
                "prediction": prediction,
                "confidence": confidence,
                "created_at": datetime.now()
            })

            st.session_state["last_prediction"] = {
                "prediction": prediction,
                "confidence": confidence
            }
            st.success("Prediction saved to MongoDB!")

    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]

        if result["prediction"] == "spam":
            st.error("🚨 Prediction: SPAM")
        else:
            st.success("✅ Prediction: NOT SPAM")

        st.write(f"Model confidence: {result['confidence']:.2f}%")

    st.divider()

    # ---------------- READ ----------------

    st.header("2. SMS History — Read")

    records = list(collection.find().sort("_id", -1))

    if records:
        display_records = []

        for r in records:
            display_records.append({
                "ID": str(r["_id"]),
                "SMS": r.get("sms_text", ""),
                "Prediction": r.get("prediction", ""),
                "Confidence (%)": r.get("confidence"),
                "Created At": r.get("created_at")
            })

        st.dataframe(
            pd.DataFrame(display_records),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No SMS records found. Predict an SMS to create your first record.")

    st.divider()

    # ---------------- UPDATE ----------------

    st.header("3. Update SMS Record")

    if records:
        record_map = {str(r["_id"]): r for r in records}
        selected_id = st.selectbox(
            "Select a record to update",
            options=list(record_map.keys()),
            format_func=lambda rid: (
                f"{record_map[rid].get('sms_text', '')[:55]} "
                f"({record_map[rid].get('prediction', '')})"
            ),
            key="update_select"
        )

        selected_record = record_map[selected_id]

        with st.form("update_form"):
            edited_sms = st.text_area(
                "Edit SMS text",
                value=selected_record.get("sms_text", ""),
                key="edited_sms"
            )

            update_button = st.form_submit_button("Update Record")

        if update_button:
            if not edited_sms.strip():
                st.warning("SMS text cannot be empty.")
            else:
                new_prediction = model.predict([edited_sms.strip()])[0]
                new_confidence = float(
                    max(model.predict_proba([edited_sms.strip()])[0]) * 100
                )

                collection.update_one(
                    {"_id": ObjectId(selected_id)},
                    {"$set": {
                        "sms_text": edited_sms.strip(),
                        "prediction": new_prediction,
                        "confidence": new_confidence,
                        "updated_at": datetime.now()
                    }}
                )

                st.success("Record updated successfully!")
                st.rerun()
    else:
        st.info("No records available to update.")

    st.divider()

    # ---------------- DELETE ----------------

    st.header("4. Delete SMS Record")

    if records:
        delete_map = {str(r["_id"]): r for r in records}

        delete_id = st.selectbox(
            "Select a record to delete",
            options=list(delete_map.keys()),
            format_func=lambda rid: (
                f"{delete_map[rid].get('sms_text', '')[:55]} "
                f"({delete_map[rid].get('prediction', '')})"
            ),
            key="delete_select"
        )

        confirm_delete = st.checkbox(
            "I confirm that I want to delete this record."
        )

        if st.button("Delete Selected Record", type="secondary"):
            if confirm_delete:
                collection.delete_one({"_id": ObjectId(delete_id)})
                st.success("Record deleted successfully!")
                st.rerun()
            else:
                st.warning("Please confirm before deleting.")
    else:
        st.info("No records available to delete.")

except Exception as error:
    st.error(f"Application error: {error}")
    st.info(
        "Check that MongoDB is running, pymongo is installed, "
        "and spam.csv exists in the project folder."
    )
