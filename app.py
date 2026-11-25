"""
ISOM5160 Homework 2 — SMS Spam / Smishing Alert App
Course: ISOM5160-L3 Python Programming for Business Applications, Fall 2026 (PG ICSM)

INSTRUCTIONS (read carefully)
- Complete the marked STUDENT sections and the function bodies below.
- Keep these function names: load_model, load_test_samples, classify_text,
  compute_accuracy, main.
- Call main() at the bottom of this file (the starter already does this).
- In GitHub and on Streamlit Cloud, this file must be named app.py.
- Submit to Canvas: a zip file containing app.py, requirements.txt, and
  Homework2_<StudentID>_link.txt (Streamlit Cloud URL and GitHub URL).
  Put those three files at the root of the zip, not inside a subfolder.
- Do NOT fine-tune a model. Do NOT scrape the web.
- Do NOT change DATASET_ID, MODEL_ID, N_TEST, or SPLIT_SEED.
- When you deploy, open Advanced settings and set Python to 3.12.
  If the app is already deployed, delete it and redeploy — the Python
  version cannot be changed in place.
- If the app calls a Hugging Face model, store a fine-grained Hugging Face
  API token in Streamlit Cloud secrets as HF_TOKEN. Do not put the token
  in app.py, requirements.txt, or any public file.
- First run may download the model and dataset (needs network). CPU is OK.

SECRET PHRASE (must appear on the app page): ISOM5160-HW2-SMS
"""

# =============================================================================
# STUDENT ANSWERS BLOCK (fill in; keep this in the SAME .py file)
# =============================================================================
# Student ID: 21329863
#
# GenAI acknowledgment (required if you used GenAI; otherwise write "None"):
#   - Tools used: OpenAI Codex (ChatGPT).
#   - How you used them: Completed the starter functions, drafted the business
#     answers, and debugged and tested the app with the required model/dataset.
#
# Q1. Who is the primary business user of this app?
#     (one short sentence; e.g., an employee or a security analyst reviewing
#      inbound SMS for spam or scam-like messages)
#   A security analyst reviews inbound business SMS messages for spam or scams.
#
# Q2. Look at the accuracy from your 100-sample test. Is accuracy enough
#     for a business to rely on this app, and why or why not?
#     2–4 short sentences max. You MUST cite the accuracy figure from
#     YOUR app run. Comment on the cost of a false positive (blocking a
#     legitimate SMS, such as a bank or one-time code message) versus a
#     false negative (missing a spam or scam-like message).
#   My app achieved 99.00% accuracy (99/100) on the fixed 100-sample test.
#   This small, fixed sample is insufficient evidence for business reliance,
#   so the app should support human review.
#   A false positive could block a bank alert or one-time code, while a false
#   negative could expose the user to a scam and financial loss.
#
# =============================================================================

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from datasets import load_dataset
from transformers import pipeline

# Locked by the instructor — do not change these four values.
DATASET_ID = "ucirvine/sms_spam"
MODEL_ID = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
N_TEST = 100
SPLIT_SEED = 42

# Course identity (required)
STUDENT_ID = "21329863"  # STUDENT: replace with your real student ID
COURSE_SECRET = "ISOM5160-HW2-SMS"  # do not change


def map_label(raw_label):
    """Map model or dataset labels to the two display names used in this app."""
    text = str(raw_label).strip().lower()
    if text in {"ham", "legitimate", "label_0", "0"}:
        return "legitimate"
    if text in {"spam", "smishing", "label_1", "1"}:
        return "spam"
    return str(raw_label)


@st.cache_resource
def load_model(token):
    """Load the SMS text-classification model once and reuse it."""
    # This function is provided. Keep device="cpu" and
    # model_kwargs={"low_cpu_mem_usage": False} so Streamlit Cloud can load
    # the model on CPU.
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        token=token,
        device="cpu",
        truncation=True,
        max_length=512,
        model_kwargs={"low_cpu_mem_usage": False},
    )
    return classifier


@st.cache_data
def load_test_samples():
    """
    Load the Hugging Face dataset and take a fixed set of 100 samples.

    The Hub copy has only a train split, so this homework holds out 100
    rows after a shuffle with SPLIT_SEED. Do not change the seed or N_TEST,
    so every student is tested on the same 100 messages.
    """
    raw = load_dataset(DATASET_ID, split="train")
    picked = raw.shuffle(seed=SPLIT_SEED).select(range(N_TEST))
    texts = list(picked["sms"])
    true_labels = [map_label(label) for label in picked["label"]]
    return texts, true_labels


def classify_text(classifier, text):
    """Return (display_label, confidence) for one SMS string."""
    prediction = classifier(text)[0]
    display_label = map_label(prediction["label"])
    score = float(prediction["score"])
    return display_label, score


def compute_accuracy(pred_labels, true_labels):
    """Return accuracy as a float between 0 and 1."""
    if len(pred_labels) != len(true_labels):
        raise ValueError("Predictions and ground truth must have the same length.")
    if not true_labels:
        raise ValueError("At least one sample is required to compute accuracy.")
    matches = sum(pred == true for pred, true in zip(pred_labels, true_labels))
    accuracy = matches / len(true_labels)
    return accuracy


def main():
    st.set_page_config(page_title="ISOM5160 - Homework 2", layout="wide")
    st.title("ISOM5160")
    st.write("Homework 2 — SMS Spam / Smishing Alert")
    st.write(f"**Student ID:** {STUDENT_ID}")
    st.write(f"**Course secret:** {COURSE_SECRET}")
    if STUDENT_ID == "YOUR_STUDENT_ID":
        st.warning("Replace YOUR_STUDENT_ID in app.py with your real student ID.")

    # Use a Hugging Face fine-grained token from Streamlit Cloud secrets
    # if one is set. Do not put the token in this file.
    try:
        hf_token = st.secrets.get("HF_TOKEN")
    except StreamlitSecretNotFoundError:
        # Public downloads also work locally without a secrets.toml file.
        hf_token = None

    try:
        with st.spinner("Loading the SMS classification model..."):
            classifier = load_model(hf_token)
    except Exception:
        st.error(
            "Could not load the model. Check the internet connection, installed "
            "dependencies, and HF_TOKEN in Streamlit secrets, then rerun the app."
        )
        st.stop()

    alert_text = st.text_area(
        "SMS message",
        value=(
            "WINNER!! You have been selected to receive a $900 prize. "
            "Claim now at http://prize-alert.example or the offer expires."
        ),
        height=140,
    )

    if st.button("Classify SMS"):
        if not alert_text.strip():
            st.warning("Please enter an SMS message before classifying it.")
        else:
            try:
                with st.spinner("Classifying the SMS..."):
                    label, confidence = classify_text(classifier, alert_text)
                st.write(f"**Prediction:** {label}")
                st.write(f"**Confidence:** {confidence:.2%}")
            except Exception:
                st.error("Could not classify this SMS. Please try again.")

    st.subheader("Test on 100 Hugging Face samples")
    st.write(
        "Click the button to load 100 messages from the Hugging Face "
        "dataset and measure accuracy."
    )
    if st.button("Run 100-sample test"):
        st.session_state.pop("test_result", None)
        progress = st.progress(0, text="Loading the 100 test messages...")
        try:
            texts, true_labels = load_test_samples()
            pred_labels = []
            rows = []
            for index, (text, true_label) in enumerate(zip(texts, true_labels), 1):
                label, confidence = classify_text(classifier, text)
                pred_labels.append(label)
                short_text = " ".join(text.split())
                if len(short_text) > 100:
                    short_text = short_text[:97] + "..."
                rows.append({
                    "#": index,
                    "Text (short)": short_text,
                    "Ground truth": true_label,
                    "Prediction": label,
                    "Confidence": f"{confidence:.2%}",
                    "Match": label == true_label,
                })
                progress.progress(index / N_TEST, text=f"Tested {index}/{N_TEST} messages")

            accuracy = compute_accuracy(pred_labels, true_labels)
            # Keep the table visible on later reruns without repeating inference.
            st.session_state["test_result"] = {"accuracy": accuracy, "rows": rows}
        except Exception:
            st.error(
                "Could not complete the 100-sample test. Check the internet "
                "connection and Hugging Face access, then try again."
            )
        finally:
            progress.empty()

    if "test_result" in st.session_state:
        result = st.session_state["test_result"]
        st.metric("Accuracy", f"{result['accuracy']:.2%}")
        correct = sum(row["Match"] for row in result["rows"])
        st.write(f"Correct predictions: {correct} / {N_TEST}")
        st.dataframe(
            result["rows"],
            hide_index=True,
            width="stretch",
            column_config={"Text (short)": st.column_config.TextColumn(width="medium")},
        )
        st.caption(
            "Accuracy is measured on the fixed homework sample; it does not "
            "establish performance on future business messages."
        )


if __name__ == "__main__":
    main()
