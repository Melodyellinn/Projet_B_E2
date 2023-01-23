
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import imblearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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

### define predict def ###   
def predict(prediction):
    y_pred = model.predict(X_test)
    return prediction

#     # if y_pred == 0:
#     #     y_pred = 'Non Faillite'
#     # elif y_pred == 1:
#     #     y_pred = 'Attention risque de Faillite'

#conf_mat = confusion_matrix(y_test, predict)
#sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Reds")
#plt.xlabel("Predicted")
#plt.ylabel("Actual")

model_prediction = pd.DataFrame()
model_prediction['Real_Value'] = y_test
model_prediction['Prediction_Value'] = predict

### APP FRONT ###
row_1_margin_1, row_1_col_1, row_2_col_2, row_1_margin_2 = st.columns((.2, 4.5, 2.5,.2))
with row_1_col_1:
    st.title('Bankrupt or not Bankrupt ?')
    if st.button('Predict'):
        pred = predict
        st.success(f'The predictions were successful')
        st.dataframe(model_prediction)

# with row_2_col_2:
#     st.title('Confusion Matrix')
#     st.write('test')
#     st.plotly_chart(conf_mat, use_container_width=False)