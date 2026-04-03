import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from io import BytesIO

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Student Analytics System", layout="wide")

# =========================
# UI DESIGN
# =========================
st.markdown("""
<style>
body {background-color: #f0f2f6;}
.card {
    padding: 20px;
    border-radius: 15px;
    background: white;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    text-align:center;
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
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_excel("student_performance.xlsx", sheet_name="Students_Data")
    users = pd.read_excel("student_performance.xlsx", sheet_name="Users")

    df['Extra_Activities'] = df['Extra_Activities'].map({'Yes': 1, 'No': 0}).fillna(0)
    df['Previous_Result'] = df['Previous_Result'].map({'A':3,'B':2,'C':1,'Fail':0}).fillna(0)
    df['Final_Result_Num'] = df['Final_Result'].map({'A':3,'B':2,'C':1,'Fail':0}).fillna(0)

    X = df[['Attendence','Study_Hours','Internal_Marks','Assignment_Score','Previous_Result','Extra_Activities']]
    y = df['Final_Result_Num']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train,y_train)

    acc = model.score(X_test,y_test)

    return df, users, model, acc

df, users, model, acc = load_data()

# =========================
# SESSION
# =========================
if "login" not in st.session_state:
    st.session_state.login = False
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# LOGIN
# =========================
st.sidebar.title("🔐 Login")

if not st.session_state.login:
    u = st.sidebar.text_input("Username")
    p = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        match = users[(users['Username']==u) & (users['Password']==p)]
        if not match.empty:
            st.session_state.login = True
            st.session_state.role = match.iloc[0]['Role']
            st.rerun()
        else:
            st.sidebar.error("Invalid Login ❌")
else:
    st.sidebar.success(f"{st.session_state.role}")
    st.sidebar.write(f"Model Accuracy: {acc:.2f}")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

# =========================
# MAIN
# =========================
if st.session_state.login:

    st.title("🎓 Student Performance System")

    # Metrics
    c1,c2,c3 = st.columns(3)
    c1.markdown(f"<div class='card'>Total Students<br><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'>Pass<br><h2>{(df['Final_Result_Num']>0).sum()}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'>Fail<br><h2>{(df['Final_Result_Num']==0).sum()}</h2></div>", unsafe_allow_html=True)

    # =========================
    # TEACHER ANALYTICS
    # =========================
    if st.session_state.role.lower() == "teacher":
        st.subheader("📊 Analytics")

        col1,col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.scatter(df['Attendence'], df['Final_Result_Num'])
            ax.set_title("Attendance vs Result")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            df['Final_Result'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

    # =========================
    # PREDICTION
    # =========================
    st.subheader("🧠 Prediction")

    att = st.number_input("Attendence",0,100,70)
    hrs = st.number_input("Study Hours",0,24,5)
    marks = st.number_input("Marks",0,100,50)
    assign = st.number_input("Assignment",0,100,50)
    prev = st.selectbox("Previous Result",["A","B","C","Fail"])
    ex = st.selectbox("Extra Activities",["Yes","No"])

    if st.button("Predict"):
        prev_val = {'A':3,'B':2,'C':1,'Fail':0}[prev]
        ex_val = 1 if ex=="Yes" else 0

        input_data = pd.DataFrame([[att,hrs,marks,assign,prev_val,ex_val]],
        columns=['Attendence','Study_Hours','Internal_Marks','Assignment_Score','Previous_Result','Extra_Activities'])

        pred = model.predict(input_data)[0]
        grade_map = {3:'A',2:'B',1:'C',0:'Fail'}

        # Recommendation
        def suggest():
            if att<60: return "Improve attendance"
            elif hrs<3: return "Increase study hours"
            elif marks<40: return "Focus on basics"
            else: return "Good work"

        if pred==0:
            st.error(f"Fail ❌ | {suggest()}")
        else:
            st.success(f"{grade_map[pred]} ✅ | {suggest()}")

        st.session_state.history.append({
            "Attendance":att,"Marks":marks,"Grade":grade_map[pred]
        })

    # =========================
    # HISTORY + DOWNLOAD
    # =========================
    if st.session_state.history:
        st.subheader("📋 History")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df)

        output = BytesIO()
        hist_df.to_excel(output,index=False)

        st.download_button("Download Report", output.getvalue(), "report.xlsx")

    # =========================
    # ADMIN PANEL
    # =========================
    if st.session_state.role.lower() == "admin":

        st.subheader("🧑‍💼 Admin Panel")

        tab1,tab2 = st.tabs(["Add","Delete"])

        with tab1:
            name = st.text_input("Name")
            att = st.number_input("Attendance",0,100)
            hrs = st.number_input("Hours",0,24)
            marks = st.number_input("Marks",0,100)
            assign = st.number_input("Assignment",0,100)
            prev = st.selectbox("Prev",["A","B","C","Fail"])
            ex = st.selectbox("Extra",["Yes","No"])
            result = st.selectbox("Result",["A","B","C","Fail"])
            perf = st.number_input("Performance Index",0,100)

            if st.button("Add Student"):
                new = {
                    "Student_ID":len(df)+1,
                    "Name":name,
                    "Attendence":att,
                    "Study_Hours":hrs,
                    "Internal_Marks":marks,
                    "Assignment_Score":assign,
                    "Previous_Result":prev,
                    "Extra_Activities":ex,
                    "Final_Result":result,
                    "Performance_Index":perf
                }

                df.loc[len(df)] = new

                with pd.ExcelWriter("student_performance.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    df.to_excel(writer, sheet_name="Students_Data", index=False)

                st.success("Added ✅")

        with tab2:
            student = st.selectbox("Select Student", df["Name"].unique())

            if st.button("Delete"):
                df = df[df["Name"]!=student]

                with pd.ExcelWriter("student_performance.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    df.to_excel(writer, sheet_name="Students_Data", index=False)

                st.success("Deleted ❌")
                st.rerun()
