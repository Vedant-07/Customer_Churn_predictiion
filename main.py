import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('Churn_pred')

# Define the app
def main():
    # Add a title
    st.title('Customer Churn Prediction')

    # Add a description
    st.write('Enter the customer information below to predict whether they will churn or not.')

    # Define the input fields
    credit_score = st.slider('Credit Score', 0, 1000, 500)
    age = st.slider('Age', 18, 100, 30)
    tenure = st.slider('Tenure', 0, 15, 5)
    balance = st.slider('Balance', 0, 250000, 50000)
    num_products = st.slider('Number of Products', 1, 4, 2)
    has_credit_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
    is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])
    estimated_salary = st.slider('Estimated Salary', 0, 300000, 100000)
    geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    
    if(geography=="France"):
        gg=0
        gs=0
    elif(geography=="Spain"):
        gg=0
        gs=1
    else:
        gg=1
        gs=0

    # Convert the input data to a dataframe
    data = pd.DataFrame({
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': 1 if has_credit_card == 'Yes' else 0,
        'IsActiveMember': 1 if is_active_member == 'Yes' else 0,
        'EstimatedSalary': estimated_salary,
        'Geography_Germany': gg,
        'Geography_France': gs,
        'Gender_Male': 0 if gender=='Female' else 1
    }, index=[0])


    pred_probs = model.predict_proba(data)[0]

    churn_prob = pred_probs[1] * 100
    not_churn_prob = pred_probs[0] * 100

    if st.button('Predict'):
        if churn_prob <= 60:
            st.warning('This customer is likely to churn.')
        else:
            st.success('This customer is not likely to churn.')

        st.write('The probability of churn is: {:.2f}%'.format(not_churn_prob))
        st.write('The probability of not churning is: {:.2f}%'.format(churn_prob))

            
if __name__ == '__main__':
    main()
