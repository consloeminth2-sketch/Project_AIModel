import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_model():
    with open('model/scaler_cancer.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model/livercancer_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, model
scaler, model = load_model()

st.set_page_config(page_title="Health Prediction App", layout="centered")

st.title("ระบบประเมินความเสี่ยงสุขภาพด้านมะเร็งตับ")
st.write("กรุณากรอกข้อมูลด้านล่างเพื่อทำการทำนายผล")

st.divider() 

col1, col2 = st.columns(2)

with col1:
    st.subheader("ข้อมูลส่วนตัวและ")
    age = st.slider("อายุ (ปี)", min_value=18, max_value=100, value=30)
    gender = st.selectbox("เพศ", ["ชาย", "หญิง"])
    hepatitis_B_map = {"เป็นบวก":"Positive", "เป็นลบ":"์Negative",}
    hepatitis_B_label = st.selectbox("ไวรัสตับอักเสบ B", list(hepatitis_B_map.keys()))
    obesity_map = {"น้ำหนักต่ำกว่าเกณฑ์":"Underweight", "น้ำหนักปกติ":"Normal", "อ้วน":"Obese", "น้ำหนักเกินเกณฑ์":"Overweight"}
    obesity_label = st.selectbox("โรคอ้วน", list(obesity_map.keys()))
with col2:
    st.subheader("พฤติกรรมการใช้ชีวิต")
    alcohol_map = {"ไม่ดื่ม / ดื่มน้อย": "Low", "ดื่มปานกลาง":"Moderate", "ดื่มหนัก":"High",}
    alcohol_label = st.selectbox("ระดับการดื่มแอลกอฮอล์", list(alcohol_map.keys()))
    smoke_map = {"ไม่สูบบุหรี่":"Non-Smoker", "สูบบุหรี่":"Smoker",}
    smoke_label = st.selectbox("สถานะการสูบบุหรี่", list(smoke_map.keys()))
    hepatitis_C_map = {"เป็นบวก":"Positive", "เป็นลบ":"์Negative",}
    hepatitis_C_label = st.selectbox("ไวรัสตับอักเสบ C", list(hepatitis_C_map.keys()))
    diabetes_map = {"เป็น":"Yes", "ไม่เป็น":"No",}
    diabetes_label = st.selectbox("โรคเบาหวาน", list(diabetes_map.keys()))

st.divider()

if st.button("ทำนายผลความเสี่ยงมะเร็งตับ", use_container_width=True):
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender_Male': [1 if gender == "ชาย" else 0],
        'Alcohol_Consumption_Low': [1 if alcohol_map[alcohol_label] == "Low" else 0],
        'Alcohol_Consumption_Moderate': [1 if alcohol_map[alcohol_label] == "Moderate" else 0],
        'Smoking_Status_Smoker': [1 if smoke_map[smoke_label] == "Smoker" else 0],
        'Hepatitis_B_Status_Positive': [1 if hepatitis_B_map[hepatitis_B_label] == "Positive" else 0 ],
        'Hepatitis_C_Status_Positive': [1 if hepatitis_C_map[hepatitis_C_label] == "Positive" else 0 ],
        'Obesity_Obese': [1 if obesity_map[obesity_label] == "Obese" else 0 ],
        'Obesity_Overweight': [1 if obesity_map[obesity_label] == "Overweight" else 0 ],
        'Obesity_Underweight': [1 if obesity_map[obesity_label] == "Underweight" else 0 ],
        'Diabetes_Yes' : [1 if diabetes_map[diabetes_label] == "Yes" else 0 ]
    })
    expected_columns = ['Age', 'Gender_Male', 'Alcohol_Consumption_Low', 'Alcohol_Consumption_Moderate', 
                        'Smoking_Status_Smoker', 'Hepatitis_B_Status_Positive', 'Hepatitis_C_Status_Positive',
                        'Obesity_Obese', 'Obesity_Overweight', 'Obesity_Underweight', 'Diabetes_Yes']
    try:
        input_data = input_data[expected_columns]
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)[0] 
        risk_percent = probabilities[1] * 100
        THRESHOLD = 42.0
        st.subheader("📊 ผลการประเมิน:")
        st.progress(int(risk_percent))
        st.write(f"โอกาสเกิดความเสี่ยงโรคตับ: **{risk_percent:.2f}%**")
        if risk_percent >= THRESHOLD:
            st.error("🚨 มีความเสี่ยงกรุณาปรึกษาแพทย์")
        else:
            st.success("✅ มีความเสี่ยงต่ำ")
    except Exception as e:
        st.warning(f"เกิดข้อผิดพลาดในการคำนวณ: รบกวนตรวจสอบว่าชื่อคอลัมน์และจำนวนคอลัมน์ตรงกับตอนเทรนโมเดลหรือไม่\n\nError: {e}")