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
# CUSTOM UI (PREMIUM LOOK)
# =========================
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("student_performance.xlsx", sheet_name="Students_Data")
users = pd.read_excel("student_performance.xlsx", sheet_name="Users")

# =========================
# PREPROCESSING
# =========================
df['Extra_Activities'] = df['Extra_Activities'].map({'Yes': 1, 'No': 0})
df['Final_Result'] = df['Final_Result'].map({'A': 3, 'B': 2, 'C': 1, 'Fail': 0})

X = df[['Attendence', 'Study_Hours', 'Internal_Marks', 'Assignment_Score', 'Extra_Activities']]
y = df['Final_Result']

model = RandomForestClassifier()
model.fit(X, y)

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
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        # Clean Excel data
        users['Username'] = users['Username'].astype(str).str.strip()
        users['Password'] = users['Password'].astype(str).str.strip()

        # Clean input
        username = username.strip()
        password = password.strip()

        # Match user
        user = users[
        (users['Username'] == username) & 
        (users['Password'] == password)
        ]
        if not user.empty:
            st.session_state.logged_in = True
            st.session_state.role = user.iloc[0]['Role']
            st.success("Login Successful ✅")
        else:
            st.error("Invalid Credentials ❌")

else:
    st.sidebar.success(f"Logged in as {st.session_state.role}")
    
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.experimental_rerun()

# =========================
# MAIN UI AFTER LOGIN
# =========================
if st.session_state.logged_in:

    st.title("🎓 Student Performance Dashboard")

    # Dashboard Cards
    col1, col2, col3 = st.columns(3)
    col1.markdown(f'<div class="card">📊 Total Students<br><h2>{len(df)}</h2></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="card">✅ Pass Count<br><h2>{(df["Final_Result"]>0).sum()}</h2></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="card">❌ Fail Count<br><h2>{(df["Final_Result"]==0).sum()}</h2></div>', unsafe_allow_html=True)

    # =========================
    # TEACHER GRAPH VIEW
    # =========================
    if st.session_state.role == "teacher":
        st.subheader("📊 Analytics")

        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots()
            ax1.scatter(df['Attendence'], df['Final_Result'])
            ax1.set_title("Attendance vs Performance")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            df['Final_Result'].value_counts().plot(kind='bar', ax=ax2)
            ax2.set_title("Grade Distribution")
            st.pyplot(fig2)

    # =========================
    # INPUT FORM
    # =========================
    st.subheader("🧠 Predict Student Performance")

    with st.form("form"):
        col1, col2 = st.columns(2)

        with col1:
            Attendence = st.number_input("Attendance (%)", 0, 100)
            Study_Hours = st.number_input("Study Hours", 0, 10)

        with col2:
            Internal_Marks = st.number_input("Internal Marks", 0, 100)
            Assignment_Score = st.number_input("Assignment Score", 0, 100)

        extra = st.selectbox("Extra Activities", ["Yes", "No"])

        submit = st.form_submit_button("Predict 🚀")

    # =========================
    # PREDICTION
    # =========================
    if submit:
        extra_val = 1 if extra == "Yes" else 0

        data = pd.DataFrame([[Attendence, Study_Hours, Internal_Marks, Assignment_Score, Extra_Activities]],
                            columns=['Attendence', 'Study_Hours', 'Internal_Marks', 'Assignment_Score', 'Extra_Activities'])

        pred = model.predict(data)[0]

        grade_map = {3: 'A', 2: 'B', 1: 'C', 0: 'Fail'}
        grade = grade_map[pred]

        if pred == 0:
            risk = "High"
            suggestion = "Immediate improvement needed"
        elif pred == 1:
            risk = "Medium"
            suggestion = "Increase study hours"
        else:
            risk = "Low"
            suggestion = "Good performance"

        # Store history
        st.session_state.history.append({
            "Attendence": Attendence,
            "Study_Hours": Study_Hours,
            "Marks": Internal_Marks,
            "Assignment": Assignment_Score,
            "Extra_Activities": Extra_Activities,
            "Grade": grade,
            "Risk": risk
        })

        st.success(f"🎓 Grade: {grade}")
        st.warning(f"⚠ Risk Level: {risk}")
        st.info(f"💡 {suggestion}")

    # =========================
    # HISTORY + DOWNLOAD
    # =========================
    st.subheader("📊 Prediction History")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)

        # Convert to Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            history_df.to_excel(writer, index=False)

        st.download_button(
            "📥 Download Excel",
            data=output.getvalue(),
            file_name="prediction_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.write("No data yet")
