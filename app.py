import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from io import BytesIO

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Student Predictor", layout="wide")

# =========================
# CUSTOM UI
# =========================
st.markdown("""
<style>
.main { background-color: #f5f7fa; }
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    text-align: center;
    color: #333;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD & TRAIN (Cached for speed)
# =========================
@st.cache_data
def load_and_train():
    # Note: Ensure these files exist in your directory
    df = pd.read_excel("student_performance.xlsx", sheet_name="Students_Data")
    users = pd.read_excel("student_performance.xlsx", sheet_name="Users")
    
    # Preprocessing
    df['Extra_Activities'] = df['Extra_Activities'].map({'Yes': 1, 'No': 0}).fillna(0)
    df['Final_Result_Num'] = df['Final_Result'].map({'A': 3, 'B': 2, 'C': 1, 'Fail': 0}).fillna(0)

    X = df[['Attendence', 'Study_Hours', 'Internal_Marks', 'Assignment_Score', 'Extra_Activities']]
    y = df['Final_Result_Num']

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return df, users, model

try:
    df, users, model = load_and_train()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# =========================
# SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# LOGIN SYSTEM
# =========================
st.sidebar.title("🔐 Login")

if not st.session_state.logged_in:
    username_input = st.sidebar.text_input("Username")
    password_input = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        users['Username'] = users['Username'].astype(str).str.strip()
        users['Password'] = users['Password'].astype(str).str.strip()
        
        user_match = users[(users['Username'] == username_input.strip()) & 
                           (users['Password'] == password_input.strip())]
        
        if not user_match.empty:
            st.session_state.logged_in = True
            st.session_state.role = user_match.iloc[0]['Role']
            st.rerun()
        else:
            st.sidebar.error("Invalid Credentials ❌")
else:
    st.sidebar.success(f"Logged in as {st.session_state.role}")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

# =========================
# MAIN UI
# =========================
if st.session_state.logged_in:
    st.title("🎓 Student Performance Dashboard")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.markdown(f'<div class="card">📊 Total Students<br><h2>{len(df)}</h2></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="card">✅ Pass Count<br><h2>{(df["Final_Result_Num"] > 0).sum()}</h2></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="card">❌ Fail Count<br><h2>{(df["Final_Result_Num"] == 0).sum()}</h2></div>', unsafe_allow_html=True)

    # Analytics for Teachers
    if st.session_state.role.lower() == "teacher":
        st.subheader("📊 Analytics")
        c1, c2 = st.columns(2)
        with c1:
            fig1, ax1 = plt.subplots()
            ax1.scatter(df['Attendence'], df['Final_Result_Num'], alpha=0.5)
            ax1.set_xlabel("Attendance %")
            ax1.set_ylabel("Grade Level")
            st.pyplot(fig1)
        with c2:
            fig2, ax2 = plt.subplots()
            df['Final_Result'].value_counts().plot(kind='bar', ax=ax2, color='skyblue')
            st.pyplot(fig2)

    # Prediction Form
    st.subheader("🧠 Predict Student Performance")
    with st.form("predict_form"):
        f1, f2 = st.columns(2)
        with f1:
            att = st.number_input("Attendance (%)", 0, 100, 75)
            hrs = st.number_input("Study Hours", 0, 24, 5)
        with f2:
            marks = st.number_input("Internal Marks", 0, 100, 50)
            assign = st.number_input("Assignment Score", 0, 100, 50)
        
        ex_act = st.selectbox("Extra Activities", ["Yes", "No"])
        submitted = st.form_submit_button("Predict 🚀")

    if submitted:
        ex_val = 1 if ex_act == "Yes" else 0
        input_data = pd.DataFrame([[att, hrs, marks, assign, ex_val]],
                                  columns=['Attendence', 'Study_Hours', 'Internal_Marks', 'Assignment_Score', 'Extra_Activities'])
        
        res = model.predict(input_data)[0]
        grade_map = {3: 'A', 2: 'B', 1: 'C', 0: 'Fail'}
        res_grade = grade_map[res]
        
        # Display Logic
        if res == 0:
            st.error(f"Predicted Grade: {res_grade} | High Risk")
        else:
            st.success(f"Predicted Grade: {res_grade} | Low/Medium Risk")

        st.session_state.history.append({
            "Attendance": att, "Hours": hrs, "Marks": marks, "Grade": res_grade
        })

    # History
    if st.session_state.history:
        st.subheader("📋 History")
        st.table(pd.DataFrame(st.session_state.history))
