# ============================================================
# DADS 5001 - Streamlit
# Uber Pickups in New York City
# + cache_data API Demo
# + cache_resource AI Model Demo
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests

from transformers import pipeline


# ============================================================
# 1. TITLE
# ============================================================

st.title("Uber pickups in NYC123.")


# ============================================================
# 2. UBER DATA SOURCE
# ============================================================

DATE_COLUMN = "date/time"

DATA_URL = (
    "https://s3-us-west-2.amazonaws.com/"
    "streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)


# ============================================================
# 3. CACHE DATA - LOAD UBER DATA
# ============================================================

# st.cache_data เหมาะกับ:
# - DataFrame
# - CSV
# - JSON
# - API response
# - ผลการคำนวณที่เป็นข้อมูล

@st.cache_data
def load_data(nrows):

    data = pd.read_csv(
        DATA_URL,
        nrows=nrows
    )

    # แสดง shape ของข้อมูล
    st.write(data.shape)

    # เปลี่ยนชื่อ column เป็นตัวพิมพ์เล็ก
    lowercase = lambda x: str(x).lower()

    data.rename(
        lowercase,
        axis="columns",
        inplace=True
    )

    # แปลง date/time เป็น datetime
    data[DATE_COLUMN] = pd.to_datetime(
        data[DATE_COLUMN]
    )

    return data


# ============================================================
# 4. LOAD UBER DATA
# ============================================================

data_load_state = st.text(
    "Loading data..."
)

# โหลด 10,000 rows
data = load_data(10000)

data_load_state.text(
    "Done! (using st.cache_data)"
)


# ============================================================
# 5. SHOW RAW DATA
# ============================================================

if st.checkbox("Show raw data"):

    st.subheader("Raw data")

    st.write(data)


# ============================================================
# 6. PICKUPS BY HOUR
# ============================================================

st.subheader(
    "Number of pickups by hour"
)

hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour,
    bins=24,
    range=(0, 24)
)[0]

st.bar_chart(hist_values)


# ============================================================
# 7. FILTER BY HOUR
# ============================================================

hour_to_filter = st.slider(
    "hour",
    0,
    23,
    17
)

filtered_data = data[
    data[DATE_COLUMN].dt.hour
    ==
    hour_to_filter
]


# ============================================================
# 8. MAP
# ============================================================

st.subheader(
    f"Map of all pickups at {hour_to_filter}:00"
)

st.map(filtered_data)


# ============================================================
# 9. API DEMO
#    st.cache_data
# ============================================================

st.divider()

st.subheader(
    "🌐 API Demo - st.cache_data"
)


# ------------------------------------------------------------
# Function นี้เรียก API
#
# ใช้ cache_data เพราะสิ่งที่ return คือ JSON data
# ------------------------------------------------------------

@st.cache_data
def api_call():

    response = requests.get(
        "https://jsonplaceholder.typicode.com/posts/1",
        timeout=10
    )

    # ถ้า API error ให้หยุด
    response.raise_for_status()

    return response.json()


# เรียก API
try:

    ans = api_call()

    st.write(
        "API response:"
    )

    st.json(ans)

except Exception as e:

    st.error(
        "API request failed."
    )

    st.write(e)


# ============================================================
# 10. AI MODEL DEMO
#     st.cache_resource
# ============================================================

st.divider()

st.subheader(
    "🤖 Sentiment Analysis - st.cache_resource"
)


# ------------------------------------------------------------
# st.cache_resource เหมาะกับ:
#
# - Machine Learning Model
# - Hugging Face pipeline
# - Database connection
# - Resource ที่โหลดช้าและใช้ซ้ำ
# ------------------------------------------------------------

@st.cache_resource
def load_model():

    return pipeline(
        "text-classification",
        model="tabularisai/multilingual-sentiment-analysis"
    )


# ============================================================
# 11. LOAD MODEL
# ============================================================

with st.spinner(
    "Loading sentiment model..."
):

    model = load_model()


# ============================================================
# 12. TEXT INPUT
# ============================================================

query = st.text_input(
    "Your query",
    value="I love Streamlit!"
)


# ============================================================
# 13. SENTIMENT PREDICTION
# ============================================================

if query:

    result = model(query)[0]

    st.write(
        "Prediction result:"
    )

    st.write(result)


    # แสดงเฉพาะ Label
    st.metric(
        "Sentiment",
        result["label"]
    )


    # แสดง Confidence Score
    st.metric(
        "Confidence",
        f"{result['score']:.2%}"
    )
