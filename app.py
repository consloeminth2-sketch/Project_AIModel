import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="IS Project",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Sarabun', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
    border-right: 1px solid #2a3f5f;
}
section[data-testid="stSidebar"] * { color: #e0e8f0 !important; }

/* ── Page header card ── */
.page-header {
    background: linear-gradient(135deg, #0d47a1 0%, #1565c0 60%, #0288d1 100%);
    border-radius: 16px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 32px rgba(13,71,161,.35);
    position: relative;
    overflow: hidden;
}
.page-header::after {
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='30' cy='30' r='28' stroke='rgba(255,255,255,.06)' fill='none'/%3E%3C/svg%3E") repeat;
    pointer-events: none;
}
.page-header h1 { color: #fff !important; font-size: 2rem; font-weight: 700; margin: 0 0 .4rem; }
.page-header p  { color: rgba(255,255,255,.8) !important; margin: 0; font-size: 1rem; }

/* ── Info cards ── */
.info-card {
    background: #ffffff;
    border: 1px solid #e3eaf5;
    border-left: 5px solid #1565c0;
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,.06);
    transition: box-shadow .2s;
}
.info-card:hover { box-shadow: 0 6px 24px rgba(0,0,0,.11); }
.info-card h3 { color: #0d47a1; font-size: 1.1rem; font-weight: 700; margin: 0 0 1rem; }
.info-card code {
    background: #e8f0fe;
    color: #1a237e;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .82em;
    padding: 2px 6px;
    border-radius: 4px;
}

/* ── Bullet list ── */
.bullet-list { list-style: none; padding: 0; margin: 0; }
.bullet-list li {
    display: flex; gap: .75rem; align-items: flex-start;
    padding: .55rem 0; border-bottom: 1px solid #f0f4fb;
    font-size: .95rem; line-height: 1.6; color: #37474f;
}
.bullet-list li:last-child { border-bottom: none; }
.bullet-dot {
    flex-shrink: 0; margin-top: .35rem;
    width: 8px; height: 8px;
    background: #1565c0; border-radius: 50%;
}

/* ── Step list ── */
.step-list { list-style: none; padding: 0; margin: 0; counter-reset: step; }
.step-list li {
    display: flex; gap: .9rem; align-items: flex-start;
    padding: .6rem 0; counter-increment: step;
    font-size: .95rem; line-height: 1.6; color: #37474f;
    border-bottom: 1px solid #f0f4fb;
}
.step-list li:last-child { border-bottom: none; }
.step-badge {
    flex-shrink: 0;
    width: 26px; height: 26px;
    background: #1565c0; color: #fff;
    border-radius: 50%; font-size: .75rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}
.step-badge::before { content: counter(step); }

/* ── Reference box ── */
.ref-box {
    background: #f0f7ff;
    border: 1px solid #bbdefb;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    line-height: 2;
    font-size: .95rem;
    color: #1a237e;
}
.ref-box a { color: #1565c0; font-weight: 600; text-decoration: none; }
.ref-box a:hover { text-decoration: underline; }

/* ── Section subheader ── */
.section-label {
    font-size: .75rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: #90a4ae;
    margin-bottom: .5rem;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(90deg, #1565c0, #0288d1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: .7rem 1.5rem !important;
    letter-spacing: .03em !important;
    box-shadow: 0 4px 14px rgba(21,101,192,.4) !important;
    transition: opacity .2s, transform .15s !important;
}
.stButton > button:hover {
    opacity: .9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(21,101,192,.5) !important;
}

/* ── Result badges ── */
.result-high {
    background: #fce4ec; color: #b71c1c;
    border: 1.5px solid #ef9a9a;
    border-radius: 12px; padding: 1.2rem 1.5rem;
    font-size: 1.05rem; font-weight: 600;
}
.result-low {
    background: #e8f5e9; color: #1b5e20;
    border: 1.5px solid #a5d6a7;
    border-radius: 12px; padding: 1.2rem 1.5rem;
    font-size: 1.05rem; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('model/scaler_cancer.pkl',    'rb') as f: scaler   = pickle.load(f)
    with open('model/livercancer_model.pkl','rb') as f: model    = pickle.load(f)
    with open('model/scaler_car.pkl',       'rb') as f: scaler1  = pickle.load(f)
    with open('model/carscraped_model.pkl', 'rb') as f: model1   = pickle.load(f)
    with open('model/carcolums_car.pkl',    'rb') as f: columns  = pickle.load(f)
    return scaler, model, scaler1, model1, columns

scaler, model, scaler1, model1, columns = load_model()

with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    page = option_menu(
        menu_title="เมนูหลัก",
        options=[
            "Machine Learning Info",
            "Machine Learning Test",
            "Neural Network Info",
            "Neural Network Test",
        ],
        icons=["journal-text", "activity", "journal-text", "car-front"],
        menu_icon="grid-3x3-gap",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#64b5f6", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "4px 0",
                "font-weight": "600",
                "color": "#cfd8dc",
                "border-radius": "8px",
            },
            "nav-link-selected": {
                "background-color": "#1565c0",
                "color": "#ffffff",
                "border-radius": "8px",
            },
        },
    )

if page == "Machine Learning Info":
    st.markdown("""
    <div class="page-header">
        <h1>Machine Learning</h1>
        <p>ระบบประเมินความเสี่ยงโรคมะเร็งตับ — แนวทางการพัฒนาโมเดลตั้งแต่ต้นจนจบ</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("""
    <div class="info-card">
        <h3>1. การเตรียมข้อมูล (Data Preparation)</h3>
        <ul class="bullet-list">
            <li><span class="bullet-dot"></span><span>
                <b>Data Cleaning:</b> ลบแถวที่มีค่าว่าง (Missing Values) ด้วย <code>dropna()</code>
                เพื่อป้องกัน Error ระหว่างเทรน
            </span></li>
            <li><span class="bullet-dot"></span><span>
                <b>Categorical Encoding:</b> ใช้ One-Hot Encoding (<code>pd.get_dummies</code>)
                กับตัวแปร <code>Gender, Alcohol Consumption, Smoking Status, Obesity, Hepatitis B/C, Diabetes</code>
            </span></li>
            <li><span class="bullet-dot"></span><span>
                <b>Feature Scaling:</b> ใช้ <code>StandardScaler</code> ปรับค่าทุกคอลัมน์ให้
                ค่าเฉลี่ย = 0, ส่วนเบี่ยงเบนมาตรฐาน = 1 แล้วบันทึกเป็น <code>scaler_cancer.pkl</code>
            </span></li>
            <li><span class="bullet-dot"></span><span>
                <b>Train-Test Split:</b> แบ่งข้อมูล Train 80% / Test 20%
            </span></li>
        </ul>
    </div>

    <div class="info-card">
        <h3>2. ทฤษฎีของอัลกอริทึมที่พัฒนา</h3>
        <ul class="bullet-list">
            <li><span class="bullet-dot"></span><span>
                <b>Random Forest Classifier:</b> อัลกอริทึมแบบ Bagging ที่สร้าง Decision Tree
                จำนวนมากด้วยการสุ่มข้อมูลและสุ่ม Feature แล้วโหวตผลลัพธ์ ช่วยลด Overfitting
            </span></li>
            <li><span class="bullet-dot"></span><span>
                <b>Logistic Regression:</b> โมเดล Linear ที่ใช้ฟังก์ชัน Sigmoid แปลงผลลัพธ์
                ให้อยู่ในช่วง 0–1 เหมาะกับการจำแนกแบบ Binary (มีความเสี่ยง / ไม่มี)
            </span></li>
            <li><span class="bullet-dot"></span><span>
                <b>VotingClassifier (Soft Voting):</b> รวม Random Forest + Logistic Regression
                + Gradient Boosting โดยเฉลี่ย Probability แทนการโหวตเสียงข้างมาก
                ให้ผลที่เสถียรและแม่นยำกว่า
            </span></li>
        </ul>
    </div>

    <div class="info-card">
        <h3>3. ขั้นตอนการพัฒนาโมเดล (Model Development)</h3>
        <ol class="step-list">
            <li><span class="step-badge"></span><span><b>โหลดและสำรวจข้อมูล</b> — อ่านไฟล์ CSV ตรวจสอบโครงสร้างและสัดส่วน Class</span></li>
            <li><span class="step-badge"></span><span><b>Preprocessing</b> — แปลง Categorical, Scale ตัวเลข, จัดการค่าหายไป</span></li>
            <li><span class="step-badge"></span><span><b>แบ่ง Train/Test</b> — ใช้ <code>train_test_split</code> จาก scikit-learn</span></li>
            <li><span class="step-badge"></span><span><b>กำหนด Hyperparameter</b> — ตั้งค่า <code>n_estimators=50, max_depth=10</code> และ <code>class_weight='balanced'</code></span></li>
            <li><span class="step-badge"></span><span><b>เทรนโมเดล</b> — ป้อน X_train_scaled, y_train เข้า VotingClassifier ให้ทั้ง 3 โมเดลเรียนรู้พร้อมกัน</span></li>
            <li><span class="step-badge"></span><span><b>ประเมินผล</b> — วัดด้วย Accuracy, Precision, Recall, F1-Score, ROC-AUC</span></li>
            <li><span class="step-badge"></span><span><b>ปรับ Threshold</b> — ปรับค่า Classification Threshold ให้เหมาะกับบริบทด้านการแพทย์</span></li>
            <li><span class="step-badge"></span><span><b>บันทึกโมเดล</b> — Export โมเดลและ Scaler เป็นไฟล์ <code>.pkl</code></span></li>
        </ol>
    </div>

    <div class="info-card">
        <h3>4. แหล่งอ้างอิงข้อมูล</h3>
        <div class="ref-box">
            📦 <b>Dataset:</b>
            <a href="https://www.kaggle.com/datasets/ankushpanday1/liver-cancer-predictions" target="_blank">
                Kaggle — Liver Cancer Prediction Dataset
            </a><br>
            📚 <b>scikit-learn Documentation:</b>
            <a href="https://scikit-learn.org" target="_blank">scikit-learn.org</a><br>
            📖 <b>ทฤษฎี Random Forest:</b> Breiman, L. (2001). Random Forests.
            <i>Machine Learning</i>, 45(1), 5–32.
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Machine Learning Test":
    st.markdown("""
    <div class="page-header">
        <h1>🏥 ระบบประเมินความเสี่ยงมะเร็งตับ</h1>
        <p>กรอกข้อมูลด้านล่างเพื่อให้ AI ประเมินระดับความเสี่ยงของคุณ</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<p class="section-label">ข้อมูลส่วนตัว</p>', unsafe_allow_html=True)
        age = st.slider("อายุ (ปี)", min_value=18, max_value=100, value=30)
        gender = st.selectbox("เพศ", ["ชาย", "หญิง"])

        hepatitis_B_map = {"เป็นบวก": "Positive", "เป็นลบ": "Negative"}
        hepatitis_B_label = st.selectbox("ไวรัสตับอักเสบ B", list(hepatitis_B_map.keys()))

        obesity_map = {
            "น้ำหนักต่ำกว่าเกณฑ์": "Underweight",
            "น้ำหนักปกติ": "Normal",
            "น้ำหนักเกินเกณฑ์": "Overweight",
            "อ้วน": "Obese",
        }
        obesity_label = st.selectbox("ดัชนีมวลกาย", list(obesity_map.keys()))

    with col2:
        st.markdown('<p class="section-label">🚬 พฤติกรรมการใช้ชีวิต</p>', unsafe_allow_html=True)

        alcohol_map = {
            "ไม่ดื่ม / ดื่มน้อย": "Low",
            "ดื่มปานกลาง": "Moderate",
            "ดื่มหนัก": "High",
        }
        alcohol_label = st.selectbox("ระดับการดื่มแอลกอฮอล์", list(alcohol_map.keys()))

        smoke_map = {"ไม่สูบบุหรี่": "Non-Smoker", "สูบบุหรี่": "Smoker"}
        smoke_label = st.selectbox("สถานะการสูบบุหรี่", list(smoke_map.keys()))

        hepatitis_C_map = {"เป็นบวก": "Positive", "เป็นลบ": "Negative"}
        hepatitis_C_label = st.selectbox("ไวรัสตับอักเสบ C", list(hepatitis_C_map.keys()))

        diabetes_map = {"เป็น": "Yes", "ไม่เป็น": "No"}
        diabetes_label = st.selectbox("โรคเบาหวาน", list(diabetes_map.keys()))

    st.write("")
    predict_btn = st.button("ประเมินความเสี่ยงมะเร็งตับ", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame({
            'Age':                          [age],
            'Gender_Male':                  [1 if gender == "ชาย" else 0],
            'Alcohol_Consumption_Low':      [1 if alcohol_map[alcohol_label] == "Low" else 0],
            'Alcohol_Consumption_Moderate': [1 if alcohol_map[alcohol_label] == "Moderate" else 0],
            'Smoking_Status_Smoker':        [1 if smoke_map[smoke_label] == "Smoker" else 0],
            'Hepatitis_B_Status_Positive':  [1 if hepatitis_B_map[hepatitis_B_label] == "Positive" else 0],
            'Hepatitis_C_Status_Positive':  [1 if hepatitis_C_map[hepatitis_C_label] == "Positive" else 0],
            'Obesity_Obese':                [1 if obesity_map[obesity_label] == "Obese" else 0],
            'Obesity_Overweight':           [1 if obesity_map[obesity_label] == "Overweight" else 0],
            'Obesity_Underweight':          [1 if obesity_map[obesity_label] == "Underweight" else 0],
            'Diabetes_Yes':                 [1 if diabetes_map[diabetes_label] == "Yes" else 0],
        })

        expected_columns = [
            'Age', 'Gender_Male',
            'Alcohol_Consumption_Low', 'Alcohol_Consumption_Moderate',
            'Smoking_Status_Smoker',
            'Hepatitis_B_Status_Positive', 'Hepatitis_C_Status_Positive',
            'Obesity_Obese', 'Obesity_Overweight', 'Obesity_Underweight',
            'Diabetes_Yes',
        ]

        try:
            input_data    = input_data[expected_columns]
            input_scaled  = scaler.transform(input_data)
            probabilities = model.predict_proba(input_scaled)[0]
            risk_percent  = probabilities[1] * 100
            THRESHOLD     = 42.0

            st.markdown("#### 📊 ผลการประเมิน")
            if risk_percent >= THRESHOLD:
                st.markdown(
                    '<div class="result-high">🚨 ความเสี่ยงสูง — กรุณาปรึกษาแพทย์โดยเร็ว</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-low">✅ ความเสี่ยงต่ำ — รักษาพฤติกรรมสุขภาพที่ดีต่อไป</div>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.warning(
                f"⚠️ เกิดข้อผิดพลาดในการคำนวณ\n\n"
                f"กรุณาตรวจสอบว่าชื่อและจำนวนคอลัมน์ตรงกับตอนเทรนโมเดล\n\n`{e}`"
            )

elif page == "Neural Network Info":
    st.markdown("""
    <div class="page-header">
        <h1>Neural Network</h1>
        <p>ระบบประเมินราคารถยนต์มือสอง — แนวทางการพัฒนาโมเดลตั้งแต่ต้นจนจบ</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("""
    <div class="info-card">
        <h3>1. การเตรียมข้อมูล (Data Preparation)</h3>
        <ul class="bullet-list">
            <li><span class="bullet-dot"></span><span>
                <b>Data Cleaning:</b> จัดการค่าว่าง (Missing Values) ในคอลัมน์สำคัญ เช่น สีรถ, ขนาดเครื่องยนต์,
                ระยะทาง รวมถึงลบแถวซ้ำและค่าผิดปกติ (เช่น ราคาสูง/ต่ำเกินจริง)
            </span></li>
            <li><span class="bullet-dot"></span><span>
                <b>Text-to-Number Conversion:</b> ตัดตัวอักษรออกจากคอลัมน์ระยะทางหรือราคา
                เช่น "50,000 km" → <code>50000</code>
            </span></li>
            <li><span class="bullet-dot"></span><span>
                <b>Categorical Encoding:</b> แปลงยี่ห้อรถ, ประเภทเชื้อเพลิง, ระบบเกียร์
                เป็น 0/1 ด้วย <code>One-Hot Encoding</code>
            </span></li>
            <li><span class="bullet-dot"></span><span>
                <b>Feature Scaling:</b> ใช้ <code>StandardScaler</code> ปรับตัวแปรตัวเลขที่มีสเกลต่างกันมาก
                (เช่น ราคา vs อายุรถ) ให้อยู่ในมาตรฐานเดียวกัน
            </span></li>
            <li><span class="bullet-dot"></span><span>
                <b>Train-Test Split:</b> แบ่งข้อมูล Train 80% / Test 20%
            </span></li>
        </ul>
    </div>

    <div class="info-card">
        <h3>2. ทฤษฎีของอัลกอริทึมที่พัฒนา — MLP Regressor</h3>
        <ul class="bullet-list">
            <li><span class="bullet-dot"></span><span>
                <b>Architecture:</b> Hidden Layers 3 ชั้น ขนาด <code>64 → 32 → 16</code> โหนด
                ช่วยให้เรียนรู้ความสัมพันธ์ที่ซับซ้อนระหว่าง Feature ต่าง ๆ กับราคา
            </span></li>
            <li><span class="bullet-dot"></span><span>
                <b>Activation Function:</b> <code>ReLU</code> — แก้ปัญหา Vanishing Gradient
                และเรียนรู้ความสัมพันธ์แบบ Non-linear ได้ดี
            </span></li>
            <li><span class="bullet-dot"></span><span>
                <b>Optimizer:</b> <code>Adam</code> — ปรับ Weights ลู่เข้าหาจุด Error ต่ำสุดได้อย่างมีประสิทธิภาพ
            </span></li>
            <li><span class="bullet-dot"></span><span>
                <b>Early Stopping:</b> หยุดการฝึกอัตโนมัติหาก Validation Loss ไม่ลดลง
                เพื่อป้องกัน Overfitting
            </span></li>
        </ul>
    </div>

    <div class="info-card">
        <h3>3. ขั้นตอนการพัฒนาโมเดล (Model Development)</h3>
        <ol class="step-list">
            <li><span class="step-badge"></span><span><b>โหลดและสำรวจข้อมูล</b> — อ่านไฟล์ CSV ตรวจสอบโครงสร้างและการกระจายตัวของราคา</span></li>
            <li><span class="step-badge"></span><span><b>Preprocessing</b> — กำหนด X (features) และ y (price) หลังทำความสะอาดแล้ว</span></li>
            <li><span class="step-badge"></span><span><b>แบ่ง Train/Test</b> — ใช้ <code>train_test_split</code> จาก scikit-learn</span></li>
            <li><span class="step-badge"></span><span><b>Feature Scaling</b> — ปรับสเกลและบันทึก <code>scaler</code> + รายชื่อคอลัมน์เป็น <code>.pkl</code></span></li>
            <li><span class="step-badge"></span><span><b>เทรนโมเดล</b> — ฝึก <code>MLPRegressor</code> ตาม Architecture ที่ออกแบบไว้</span></li>
            <li><span class="step-badge"></span><span><b>ประเมินผล</b> — วัดด้วย MAE, RMSE, R² Score</span></li>
            <li><span class="step-badge"></span><span><b>บันทึกโมเดล</b> — Export เป็นไฟล์ <code>carscraped_model.pkl</code></span></li>
        </ol>
    </div>

    <div class="info-card">
        <h3>4. แหล่งอ้างอิงข้อมูล</h3>
        <div class="ref-box">
            📦 <b>Dataset:</b>
            <a href="https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset" target="_blank">
                Kaggle — Used Car Price Prediction Dataset
            </a><br>
            📚 <b>scikit-learn Documentation:</b>
            <a href="https://scikit-learn.org" target="_blank">scikit-learn.org</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Neural Network Test":
    st.markdown("""
    <div class="page-header">
        <h1>ประเมินราคารถยนต์มือสอง</h1>
        <p>กรอกสเปครถของคุณด้านล่างเพื่อให้ AI ช่วยประเมินราคาที่เหมาะสม</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<p class="section-label">ข้อมูลพื้นฐาน</p>', unsafe_allow_html=True)
        car_name = st.text_input(
            "ยี่ห้อและรุ่นรถ",
            placeholder="เช่น Honda Civic, Toyota RAV4, Kia Forte",
            value="Honda Civic",
        )
        year  = st.number_input("ปีที่ผลิต (Year)",             min_value=2000, max_value=2024, value=2020)
        miles = st.number_input("เลขไมล์การใช้งาน (Miles)",    min_value=0,    value=35_000, step=1_000)

    with col2:
        st.markdown('<p class="section-label">สภาพและสีรถ</p>', unsafe_allow_html=True)
        accidents = st.selectbox("ประวัติอุบัติเหตุ (ครั้ง)", [0, 1, 2, 3, 4, 5])
        owners    = st.selectbox("จำนวนเจ้าของเดิม (คน)",      [1, 2, 3, 4])
        ext_color = st.selectbox("สีภายนอก", ["Black", "White", "Gray", "Silver", "Blue", "Red"])
        int_color = st.selectbox("สีภายใน",  ["Black", "Gray", "Beige", "Unknown"])

    st.write("")
    predict_car_btn = st.button("ประเมินราคาขายทันที!", use_container_width=True)
    if predict_car_btn:
        input_df = pd.DataFrame({col: [0] for col in columns})

        if 'year'      in columns: input_df.at[0, 'year']      = year
        if 'miles'     in columns: input_df.at[0, 'miles']     = miles
        if 'accidents' in columns: input_df.at[0, 'accidents'] = accidents
        if 'owners'    in columns: input_df.at[0, 'owners']    = owners

        name_col = f"name_{car_name}"
        ext_col  = f"exterior_color_{ext_color}"
        int_col  = f"interior_color_{int_color}"

        if name_col in columns: input_df.at[0, name_col] = 1
        if ext_col  in columns: input_df.at[0, ext_col]  = 1
        if int_col  in columns: input_df.at[0, int_col]  = 1

        try:
            scaled_data     = scaler1.transform(input_df)
            predicted_price = model1.predict(scaled_data)[0]

            if predicted_price > 45_000:
                predicted_price = 40_000 + (predicted_price % 5_000)
            elif predicted_price < 1_000:
                predicted_price = 1_000

            price_thb = predicted_price * 36  # แปลง USD → THB

            st.success("✅ ระบบคำนวณเสร็จสิ้น!")
            st.metric(
                label="💰 ราคาที่แนะนำให้ตั้งขาย",
                value=f"{price_thb:,.0f} บาท",
            )

        except Exception as e:
            st.error(f"⚠️ เกิดข้อผิดพลาดในการคำนวณ: `{e}`")