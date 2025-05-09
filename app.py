import streamlit as st
import joblib

# Load your trained sentiment model (make sure it's trained for 3 categories: Positive, Negative, Neutral)
model = joblib.load("model.pkl")

def predict_emotion(text):
    """
    Predict the sentiment of the input text.
    The model should return 3 labels: Positive, Negative, or Neutral.
    """
    prediction = model.predict([text])[0]
    return prediction

# Streamlit UI
st.title("🧠 Emotion Decoder")
st.write("Upload a text file to analyze its sentiment (Positive, Negative, or Neutral):")

# File upload widget
uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv"])

if uploaded_file is not None:
    # Read the uploaded file's content
    if uploaded_file.type == "text/plain":
        # For text files
        text_content = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "text/csv":
        # For CSV files, we assume that text content is in the first column
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        text_content = " ".join(df.iloc[:, 0].astype(str))  # Join all rows in the first column

    # Display the uploaded file content
    st.subheader("File Content:")
    st.write(text_content)

    # Perform sentiment analysis
    if st.button("Analyze Sentiment"):
        if text_content:
            emotion = predict_emotion(text_content)

            # Display result with color coding based on sentiment
            if emotion == "Positive":
                st.success(f"Predicted Emotion: {emotion}")
            elif emotion == "Negative":
                st.error(f"Predicted Emotion: {emotion}")
            else:
                st.info(f"Predicted Emotion: {emotion}")
        else:
            st.warning("The file is empty. Please upload a valid file.")
