import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Title and Sidebar
st.set_page_config(page_title="Spam Mail Detector", layout="centered")
st.title(" Spam Mail Detection App")
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a Logistic Regression model trained on TF-IDF features to classify emails as spam or ham."
)

# Load and preprocess data
@st.cache_data
def load_data():
    return pd.read_csv("dataset/mail_data.csv", encoding="latin1")

    data['Category'] = data['Category'].map({'ham': 0, 'spam': 1}).astype(int)
    return data

mail_data = load_data()

# Split data
x = mail_data['Message']
y = mail_data['Category']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2)

# Feature extraction
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Train model
model = LogisticRegression()
model.fit(x_train_vec, y_train)

# Accuracy
train_acc = accuracy_score(model.predict(x_train_vec), y_train)
test_acc = accuracy_score(model.predict(x_test_vec), y_test)

# Display accuracy
with st.expander("Model Performance"):
    st.write(f"Training Accuracy: `{train_acc:.2f}`")
    st.write(f" Testing Accuracy: `{test_acc:.2f}`")

# Prediction interface
st.subheader(" Check Your Mail")
mail_input = st.text_area(" Enter your mail content below:")

if st.button("Predict"):
    if mail_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        mail_vec = vectorizer.transform([mail_input])
        prediction = model.predict(mail_vec)[0]
        if prediction == 1:
            st.error(" This is a spam mail.")
        else:
            st.success(" This is a ham (non-spam) mail.")

# Data Visualizations
with st.expander(" Data Insights"):
    st.markdown("#### Distribution of Spam vs Ham")
    category_counts = mail_data['Category'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(['Ham', 'Spam'], category_counts, color=['green', 'red'])
    ax1.set_ylabel("Count")
    ax1.set_title("Spam vs Ham Distribution")
    st.pyplot(fig1)

    st.markdown("#### Message Length Distribution")
    mail_data['Message_Length'] = mail_data['Message'].apply(len)
    fig2, ax2 = plt.subplots()
    sns.histplot(data=mail_data, x='Message_Length', hue='Category', bins=50, ax=ax2, palette={0: 'green', 1: 'red'})
    ax2.set_title("Message Length by Category")
    st.pyplot(fig2)

# Confusion Matrix
with st.expander(" Confusion Matrix"):
    y_pred = model.predict(x_test_vec)
    cm = confusion_matrix(y_test, y_pred)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    ax3.set_title("Confusion Matrix")
    st.pyplot(fig3)