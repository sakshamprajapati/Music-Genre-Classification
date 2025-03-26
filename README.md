# 🎬 Movie Genre Classification

A machine learning model that classifies movies into genres based on plot descriptions.  
This project uses **Natural Language Processing (NLP)** and **Machine Learning** to analyze movie plots and predict their genres.

🚀 **Live Demo:** [Click Here to Try the App](https://your-streamlit-app-url)  

---

## 📌 Project Objective
- Develop a model that classifies movies into genres.
- Work with text-based movie plot descriptions.
- Convert raw text into numerical vectors using NLP techniques.
- Experiment with various classifiers to determine the best fit.
- Deliver an accurate model with insights into misclassifications.

---

## 🏗️ Tech Stack
- **Python** (Programming Language)
- **Streamlit** (Frontend UI & Deployment)
- **scikit-learn** (Machine Learning)
- **NLTK** (Natural Language Processing)
- **Pandas, NumPy** (Data Handling)

---

## 🚀 How to Run the Project Locally

### 1️⃣ **Clone the Repository**
```
git clone https://github.com/YOUR_USERNAME/movie-genre-classification.git
cd movie-genre-classification
```
### 2️⃣ **Create a Virtual Environment (Optional)**
```
python -m venv env
source env/bin/activate  # On macOS/Linux
env\Scripts\activate     # On Windows
```

### 3️⃣ **Install Dependencies**
```
pip install -r requirements.txt
```

### 4️⃣ **Train the Model (If Not Already Trained)**
```
python preprocess.py
```

### 5️⃣ **Run the Streamlit App**
```
streamlit run app.py
```

---

## 🌍 Try the Live Deployed App
We have deployed this project using **Streamlit Cloud**. You can access it here:  
🔗 **Live Demo:** [Click Here to Try the App](https://your-streamlit-app-url)

---

## 📂 Project Structure
```
movie-genre-classification/
│── train_data.txt       # Training dataset
│── test_data.txt        # Test dataset
│── preprocess.py        # Data preprocessing & model training
│── app.py               # Streamlit UI for predictions
│── genre_model.pkl      # Trained ML model
│── vectorizer.pkl       # Saved TF-IDF vectorizer
│── requirements.txt     # Required Python packages
│── README.md            # Project documentation
```

---

## 🔬 Model Training
1. **Data Preprocessing**:  
   - Cleaning text (lowercasing, removing special characters, stopwords)
   - Tokenization & Lemmatization
   - Converting text to numerical vectors (TF-IDF)
  
2. **Model Selection**:  
   - Compared classifiers: Logistic Regression, SVM, Random Forest  
   - Selected the best-performing model **(Logistic Regression)**  

3. **Evaluation Metrics**:  
   - **Accuracy, Precision, Recall, F1-score**
         ✅ Accuracy: 91%
         ✅ Precision: 89%
         ✅ Recall: 90%
         ✅ F1 Score: 89%
   - Identified common misclassifications and improved feature extraction.

---
