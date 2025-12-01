import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

# Load data
mail_data = pd.read_csv(r"C:\Users\DELL 3520\OneDrive\Desktop\streamlit\dataset\mail_data.csv", encoding = 'latin1')

#Encode label
mail_data.value_counts('Category')
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 1
mail_data['Category'] = mail_data['Category'].astype('int')

#spilt data
x = mail_data['Message']
y = mail_data['Category']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.35, random_state = 2)

#Feature extraction 
feature_extraction = TfidfVectorizer()
feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)
x_train_ex = feature_extraction.fit_transform(x_train)
x_test_ex = feature_extraction.transform(x_test)

#Train model
model = LogisticRegression()
model.fit(x_train_ex, y_train)


x_train_ex_pred = model.predict(x_train_ex)
x_train_ex_accuracy = accuracy_score(x_train_ex_pred, y_train)
x_test_ex_pred = model.predict(x_test_ex)
x_test_ex_accuracy = accuracy_score(x_test_ex_pred, y_test)

#streamlit interface
st.title("spam mail detection")
mail = st.text_area("Enter your mail here")
if st.button("predict"):
    mail_transform = feature_extraction.transform([mail])
    prediction = model.predict(mail_transform)
    if prediction [0]== 1:
        st.error("This is a spam mail")
    else:
        st.success("This is ham mail")


 # Display accuracy
with st.expander(" Model Performance"):
    st.write(f" Training Accuracy: `{x_train_ex_accuracy:.2f}`")
    st.write(f" Testing Accuracy: `{x_test_ex_accuracy:.2f}`")