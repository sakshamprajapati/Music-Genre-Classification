import streamlit as st
import joblib

# Load trained model & vectorizer
model = joblib.load('genre_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Custom Styling
st.markdown(
    """
    <style>
    /* Background */
    .main {
        background-color: #121212;
        color: black;
        font-family: 'Arial', sans-serif;
    }

    }

    /* Input Box */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid #ff9800;
        border-radius: 12px;
        color: white;
        padding: 15px;
        font-size: 16px;
        box-shadow: 0px 0px 10px rgba(255, 152, 0, 0.5);
    }
    
    /* Button */
    .stButton button {
        background: linear-gradient(45deg, #ff9800, #ff5722);
        color: white;
        border-radius: 12px;
        font-size: 20px;
        font-weight: bold;
        padding: 12px;
        border: none;
        transition: all 0.3s ease-in-out;
    }

    .stButton button:hover {
        background: linear-gradient(45deg, #ff5722, #ff9800);
        transform: scale(1.05);
    }

    /* Prediction Result */
    .prediction {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        padding: 15px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.1);
        box-shadow: 0px 0px 10px rgba(255, 152, 0, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom Title with Larger Font
st.markdown('<p style="font-size:50px; font-weight: bold;text-align: center;color: #ff9800;padding: 20px 0; text-shadow: 3px 3px 8px rgba(255, 152, 0, 0.7)">🎬 Movie Genre Predictor 🎭</p>', unsafe_allow_html=True)

# Description
st.write("💡 **Enter a movie plot below, and our AI will predict its genre!**")

# User Input
user_input = st.text_area("📝 **Enter movie description:**", "")

# Predict Button
if st.button("🔍 Predict Genre"):
    if user_input:
        # Preprocess input
        user_input_clean = vectorizer.transform([user_input])
        
        # Predict genre
        prediction = model.predict(user_input_clean)[0]

        # Genre Icons
        genre_icons = {
            "Action": "🔥",
            "Comedy": "😂",
            "Drama": "🎭",
            "Horror": "👻",
            "Sci-Fi": "🚀",
            "Thriller": "🔪",
            "Romance": "❤️",
            "Fantasy": "🧙‍♂️",
            "Mystery": "🕵️",
        }
        icon = genre_icons.get(prediction, "🎬")

        # Display Prediction
        st.markdown(f'<p class="prediction">Predicted Genre: {icon} {prediction}</p>', unsafe_allow_html=True)
    else:
        st
