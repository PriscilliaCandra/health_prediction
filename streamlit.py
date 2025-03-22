import streamlit as st
import pandas as pd
import joblib

diabetes_model = joblib.load('model.pkl')  
heart_disease_model = joblib.load('model2.pkl') 

st.title('Health Prediction App')

prediction_type = st.sidebar.selectbox('Select Prediction Type', ['Diabetes Prediction', 'Heart Disease Prediction'])

if prediction_type == 'Diabetes Prediction':
    st.header('Diabetes Prediction')

    def user_input_features_diabetes():
        gender = st.selectbox('Gender', ['Female', 'Male'])
        age = st.slider('Age', 0, 100, 50)
        hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
        heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
        smoking_history = st.selectbox('Smoking History', ['never', 'No Info', 'current', 'former', 'ever', 'not current'])
        bmi = st.slider('BMI', 10.0, 60.0, 25.0)
        HbA1c_level = st.slider('HbA1c Level', 3.5, 9.0, 5.7)
        blood_glucose_level = st.slider('Blood Glucose Level', 80, 300, 140)
  
        hypertension = 1 if hypertension == 'Yes' else 0
        heart_disease = 1 if heart_disease == 'Yes' else 0
        
        data = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'smoking_history': smoking_history,
            'bmi': bmi,
            'HbA1c_level': HbA1c_level,
            'blood_glucose_level': blood_glucose_level
        }
        
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features_diabetes()

    input_encoded = pd.get_dummies(input_df, columns=['gender', 'smoking_history'], drop_first=True)

    for col in diabetes_model.feature_names_in_:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
            
    input_encoded = input_encoded[diabetes_model.feature_names_in_]

    prediction = diabetes_model.predict(input_encoded)

    st.subheader('Prediction')
    st.write('Diabetes' if prediction[0] == 1 else 'No Diabetes')

elif prediction_type == 'Heart Disease Prediction':
    st.header('Heart Disease Prediction')

    def user_input_features_heart_disease():
        age = st.slider('Age', 29, 77, 50)
        sex = st.selectbox('Sex', ['Female', 'Male'])
        cp = st.selectbox('Chest Pain Type (cp)', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        trestbps = st.slider('Resting Blood Pressure (trestbps)', 94, 200, 120)
        chol = st.slider('Cholesterol (chol)', 126, 564, 200)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', ['No', 'Yes'])
        restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
        thalach = st.slider('Maximum Heart Rate Achieved (thalach)', 71, 202, 150)
        exang = st.selectbox('Exercise Induced Angina (exang)', ['No', 'Yes'])
        oldpeak = st.slider('ST Depression Induced by Exercise (oldpeak)', 0.0, 6.2, 1.0)
        slope = st.selectbox('Slope of the Peak Exercise ST Segment (slope)', ['Upsloping', 'Flat', 'Downsloping'])
        ca = st.slider('Number of Major Vessels Colored by Fluoroscopy (ca)', 0, 4, 0)
        thal = st.selectbox('Thalassemia (thal)', ['Normal', 'Fixed Defect', 'Reversible Defect'])
        
        sex = 1 if sex == 'Male' else 0
        cp = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)
        fbs = 1 if fbs == 'Yes' else 0
        restecg = ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'].index(restecg)
        exang = 1 if exang == 'Yes' else 0
        slope = ['Upsloping', 'Flat', 'Downsloping'].index(slope)
        thal = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal)
        
        data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features_heart_disease()

    prediction = heart_disease_model.predict(input_df)

    st.subheader('Prediction')
    st.write('Heart Disease' if prediction[0] == 1 else 'No Heart Disease')
