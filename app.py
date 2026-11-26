
import os
import joblib
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Milk Quality Prediction",
    page_icon="🥛",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f4fff8, #eaf7ef);
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}

.main-title {
    text-align: center;
    color: #14532d;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #52745d;
    font-size: 17px;
    margin-bottom: 30px;
}

.section-title {
    color: #166534;
    font-size: 23px;
    font-weight: bold;
}

div.stButton > button {
    background: linear-gradient(90deg, #15803d, #22a05a);
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 17px;
    font-weight: bold;
    border: none;
}

div.stButton > button:hover {
    background: #14532d;
    color: white;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

.high {
    background-color: #dcfce7;
    border: 2px solid #16a34a;
    color: #166534;
}

.medium {
    background-color: #fef9c3;
    border: 2px solid #eab308;
    color: #854d0e;
}

.low {
    background-color: #fee2e2;
    border: 2px solid #dc2626;
    color: #991b1b;
}

.footer {
    text-align: center;
    color: #64806c;
    margin-top: 35px;
}
</style>
""", unsafe_allow_html=True)


# Header
st.markdown(
    '<div class="main-title">🥛 Milk Quality Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Milk Quality Analysis System</div>',
    unsafe_allow_html=True
)

st.divider()


# Load ML Model
@st.cache_resource
def load_model():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(
        base_dir, "milk_quality_model.pkl"
    )

    data = joblib.load(model_path)

    return data["model"], data["label_encoder"]


model, label_encoder = load_model()


# Input Section
st.markdown(
    '<div class="section-title">🔬 Enter Milk Parameters</div>',
    unsafe_allow_html=True
)

st.write("Enter the milk characteristics below to predict its quality.")

with st.container(border=True):

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Physical Parameters")

        ph = st.number_input(
            "pH Level",
            min_value=0.0,
            max_value=14.0,
            value=6.6,
            step=0.1
        )

        temprature = st.number_input(
            "Temperature (°C)",
            min_value=0,
            max_value=100,
            value=40,
            step=1
        )

        colour = st.number_input(
            "Colour",
            min_value=0,
            max_value=300,
            value=255,
            step=1
        )

        turbidity = st.selectbox(
            "Turbidity",
            options=[1, 0],
            format_func=lambda x: "High" if x == 1 else "Low"
        )

    with col2:

        st.subheader("Quality Characteristics")

        taste = st.selectbox(
            "Taste",
            options=[1, 0],
            format_func=lambda x: "Good" if x == 1 else "Bad"
        )

        odor = st.selectbox(
            "Odor",
            options=[1, 0],
            format_func=lambda x: "Good" if x == 1 else "Bad"
        )

        fat = st.selectbox(
            "Fat Content",
            options=[1, 0],
            format_func=lambda x: "High" if x == 1 else "Low"
        )


# Prediction Button
st.write("")

if st.button("🔍 Predict Milk Quality"):

    input_data = pd.DataFrame(
        [[ph, temprature, colour, taste, odor, fat, turbidity]],
        columns=[
            "ph",
            "temprature",
            "colour",
            "taste",
            "odor",
            "fat",
            "turbidity"
        ]
    )

    pred_encoded = model.predict(input_data)

    pred_label = label_encoder.inverse_transform(
        pred_encoded
    )[0]

    pred_label = str(pred_label).lower()

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )

    if pred_label == "high":

        st.markdown("""
        <div class="result-card high">
            <h1>🟢 HIGH QUALITY</h1>
            <h3>Excellent Milk Quality</h3>
            <p>The model classified this sample as High Quality.</p>
        </div>
        """, unsafe_allow_html=True)

    elif pred_label == "medium":

        st.markdown("""
        <div class="result-card medium">
            <h1>🟡 MEDIUM QUALITY</h1>
            <h3>Moderate Milk Quality</h3>
            <p>The model classified this sample as Medium Quality.</p>
        </div>
        """, unsafe_allow_html=True)

    elif pred_label == "low":

        st.markdown("""
        <div class="result-card low">
            <h1>🔴 LOW QUALITY</h1>
            <h3>Poor Milk Quality</h3>
            <p>The model classified this sample as Low Quality.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error(f"Unexpected prediction label: {pred_label}")


# Footer
st.markdown("""
<div class="footer">
    Milk Quality Prediction System
    <br>
    Developed using Python, Pandas, Scikit-learn and Streamlit
</div>
""", unsafe_allow_html=True)
