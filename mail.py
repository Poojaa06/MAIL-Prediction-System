import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
mail_data = pd.read_csv(r"C:\Users\DELL 3520\OneDrive\Desktop\streamlit\dataset\mail_data.csv", encoding='latin1')

# Encode labels
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 1
mail_data['Category'] = mail_data['Category'].astype('int')

# Split data with larger test size
x = mail_data['Message']
y = mail_data['Category']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.46, random_state=2)

# TF-IDF with limited features to reduce model complexity
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, max_features=2300)
x_train_ex = vectorizer.fit_transform(x_train)
x_test_ex = vectorizer.transform(x_test)

# Logistic Regression with stronger regularization
model = LogisticRegression(C=0.6, max_iter=1000)
model.fit(x_train_ex, y_train)

# Predictions
y_train_pred = model.predict(x_train_ex)
y_test_pred = model.predict(x_test_ex)

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Streamlit UI
st.title("Spam Mail Detection App")
mail = st.text_area("Enter your mail content here:")
if st.button("Predict"):
    mail_transform = vectorizer.transform([mail])
    prediction = model.predict(mail_transform)
    if prediction[0] == 1:
        st.error(" This is a spam mail.")
    else:
        st.success("This is a ham mail.")

# Model Performance
with st.expander("Model Performance"):
    st.write(f"Training Accuracy: `{train_acc:.2f}`")
    st.write(f"Testing Accuracy: `{test_acc:.2f}`")

    # Data Visualizations
with st.expander(" Data visualization"):

    category_counts = mail_data['Category'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(['Ham', 'Spam'], category_counts, color=['skyblue', 'black'])
    ax1.set_ylabel("Count")
    ax1.set_title("Spam and Ham")
    st.pyplot(fig1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)