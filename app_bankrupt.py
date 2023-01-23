
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
import imblearn

## Import pickle & Data ##
data = pd.read_csv("data/data.csv")

with open('True_model_RandomForest.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the target and split data for train_test
y = data['Bankrupt?']
X = data.drop('Bankrupt?', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size= 0.3,
                                                    random_state=0)
### Training ###
model_train = model.fit(X_train, y_train)
### define predict def ###   
def predict(prediction):
    y_pred = model_train.predict(X_test)
    if y_pred == 'Non Faillite':
        y_pred = 0
    elif y_pred == 'Attention risque de Faillite':
        y_pred = 1

    return prediction

st.title('Bankrupt or not Bankrupt ?')

if st.button('Predict'):
    bankrupt = predict(y_test)
    st.success(f'The predicted success')

model_prediction = pd.DataFrame()
model_prediction['Real_Value'] = y_test
model_prediction['Prediction_Value'] = predict