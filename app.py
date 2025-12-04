import streamlit as st
import pandas as pd
import numpy as np
import os

# Import model pipeline yang sudah dilatih dari model.py
from model import pipe

# Path ke data (untuk ambil pilihan dropdown dan nilai default)
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "OTP_Time_Series_Master.xlsx")

# Baca data utama
df = pd.read_excel(DATA_PATH)

# Kolom target & fitur (harus sama dengan di model.py)
target_col = "OnTime Arrivals \n(%)"
feature_cols = [
    "Route",
    "Departing Port",
    "Arriving Port",
    "Airline",
    "Month",
    "Sectors Scheduled",
    "Sectors Flown",
    "Cancellations",
    "Departures On Time",
    "Arrivals On Time",
    "Departures Delayed",
    "Arrivals Delayed",
    "OnTime Departures \n(%)",
    "Cancellations \n\n(%)",
]

# Ambil nilai unik untuk fitur kategorik
routes = sorted(df["Route"].dropna().unique().tolist())
depart_ports = sorted(df["Departing Port"].dropna().unique().tolist())
arrive_ports = sorted(df["Arriving Port"].dropna().unique().tolist())
airlines = sorted(df["Airline"].dropna().unique().tolist())

# Ambil nilai rata-rata untuk inisialisasi numeric input
numeric_defaults = df[[
    "Sectors Scheduled",
    "Sectors Flown",
    "Cancellations",
    "Departures On Time",
    "Arrivals On Time",
    "Departures Delayed",
    "Arrivals Delayed",
    "OnTime Departures \n(%)",
    "Cancellations \n\n(%)",
]].mean(numeric_only=True).to_dict()

st.title("Prediksi On-Time Arrival (%) Penerbangan")
st.markdown(
    """
Aplikasi ini memprediksi persentase *On-Time Arrivals* sebuah rute penerbangan
berdasarkan data historis dan model **Random Forest** yang sudah dilatih.
"""
)

st.header("Input Fitur Penerbangan")

# Bagian input kategorik
st.subheader("Informasi Rute & Maskapai")
col1, col2 = st.columns(2)

with col1:
    route = st.selectbox("Route", routes)
    depart_port = st.selectbox("Departing Port", depart_ports)

with col2:
    arrive_port = st.selectbox("Arriving Port", arrive_ports)
    airline = st.selectbox("Airline", airlines)

# Input bulan (menggunakan date_input, hanya bulan & tahun yang dipakai)
st.subheader("Periode Penerbangan")
month_date = st.date_input("Bulan (pilih tanggal apa saja di bulan tersebut)")

# Input numerik
st.subheader("Statistik Operasional")
c1, c2 = st.columns(2)

with c1:
    sectors_scheduled = st.number_input(
        "Sectors Scheduled",
        min_value=0.0,
        value=float(numeric_defaults.get("Sectors Scheduled", 0.0)),
    )
    sectors_flown = st.number_input(
        "Sectors Flown",
        min_value=0.0,
        value=float(numeric_defaults.get("Sectors Flown", 0.0)),
    )
    cancellations = st.number_input(
        "Cancellations",
        min_value=0.0,
        value=float(numeric_defaults.get("Cancellations", 0.0)),
    )
    dep_on_time = st.number_input(
        "Departures On Time",
        min_value=0.0,
        value=float(numeric_defaults.get("Departures On Time", 0.0)),
    )
    arr_on_time = st.number_input(
        "Arrivals On Time",
        min_value=0.0,
        value=float(numeric_defaults.get("Arrivals On Time", 0.0)),
    )

with c2:
    dep_delayed = st.number_input(
        "Departures Delayed",
        min_value=0.0,
        value=float(numeric_defaults.get("Departures Delayed", 0.0)),
    )
    arr_delayed = st.number_input(
        "Arrivals Delayed",
        min_value=0.0,
        value=float(numeric_defaults.get("Arrivals Delayed", 0.0)),
    )
    ontime_dep_pct = st.number_input(
        "OnTime Departures (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(numeric_defaults.get("OnTime Departures \n(%)", 0.0)),
    )
    cancel_pct = st.number_input(
        "Cancellations (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(numeric_defaults.get("Cancellations \n\n(%)", 0.0)),
    )

st.markdown("---")

if st.button("Prediksi On-Time Arrivals (%)"):
    # Susun 1 baris data input sesuai dengan kolom di training
    input_dict = {
        "Route": route,
        "Departing Port": depart_port,
        "Arriving Port": arrive_port,
        "Airline": airline,
        "Month": pd.to_datetime(month_date),
        "Sectors Scheduled": sectors_scheduled,
        "Sectors Flown": sectors_flown,
        "Cancellations": cancellations,
        "Departures On Time": dep_on_time,
        "Arrivals On Time": arr_on_time,
        "Departures Delayed": dep_delayed,
        "Arrivals Delayed": arr_delayed,
        "OnTime Departures \n(%)": ontime_dep_pct,
        "Cancellations \n\n(%)": cancel_pct,
    }

    input_df = pd.DataFrame([input_dict])

    # Transformasi fitur waktu agar sesuai dengan model (lihat di model.py)
    input_df["Month"] = pd.to_datetime(input_df["Month"])
    input_df["month_num"] = input_df["Month"].dt.month
    input_df["year"] = input_df["Month"].dt.year
    input_df = input_df.drop(columns="Month")

    # Pastikan urutan kolom sama dengan saat training
    categorical_cols = ["Route", "Departing Port", "Arriving Port", "Airline"]
    other_cols = [
        "Sectors Scheduled",
        "Sectors Flown",
        "Cancellations",
        "Departures On Time",
        "Arrivals On Time",
        "Departures Delayed",
        "Arrivals Delayed",
        "OnTime Departures \n(%)",
        "Cancellations \n\n(%)",
        "month_num",
        "year",
    ]
    input_df = input_df[categorical_cols + other_cols]

    # Prediksi
    try:
        pred = pipe.predict(input_df)[0]
        st.success(f"Perkiraan On-Time Arrivals: {pred:.2f} %")
    except Exception as e:
        st.error(f"Terjadi error saat melakukan prediksi: {e}")

st.caption("Model: RandomForestRegressor dengan preprocessing One-Hot Encoding untuk fitur kategorik.")