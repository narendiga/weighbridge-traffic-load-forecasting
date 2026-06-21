import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import PROJECT_ROOT, load_historical_daily
from utils.maintenance import recommend_maintenance
from utils.visualization import forecast_chart


MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATHS = {
    "Hybrid TCN-BiLSTM": MODEL_DIR / "tcn-bilstm.keras",
    "Hybrid CNN-BiGRU": MODEL_DIR / "cnn-bigru.keras",
    "CNN": MODEL_DIR / "cnn.keras",
    "TCN": MODEL_DIR / "tcn.keras",
    "BiLSTM": MODEL_DIR / "bilstm.keras",
    "BiGRU": MODEL_DIR / "bigru.keras",
}

WINDOW_SIZE = 30
TARGET_COL = "jumlah_truk_diff_scaled"
MIN_FORECAST_VALUE = 0
FORECAST_STATE_KEY = "saved_forecast_result"
INDONESIAN_DAY_NAMES = {
    0: "Senin",
    1: "Selasa",
    2: "Rabu",
    3: "Kamis",
    4: "Jumat",
    5: "Sabtu",
    6: "Minggu",
}


def inverse_1d(scaler, arr):
    arr = np.asarray(arr).reshape(-1, 1)
    return scaler.inverse_transform(arr).flatten()


def model_not_found_message(model_name, model_path):
    return f"Model {model_name} tidak ditemukan pada: {model_path}"


def format_date_with_day(value):
    timestamp = pd.Timestamp(value)
    return f"{INDONESIAN_DAY_NAMES[timestamp.dayofweek]}, {timestamp:%Y-%m-%d}"


@st.cache_resource(show_spinner=False)
def _load_keras_model(model_path_text):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path_text, safe_mode=False)


def recursive_forecast(model_name, modeling_df, scaler, horizon):
    model_path = MODEL_PATHS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(model_not_found_message(model_name, model_path))
    if scaler is None:
        raise FileNotFoundError("Scaler tidak ditemukan atau tidak berhasil dibuat dari data historis.")

    forecast_base_df = modeling_df.sort_values("tanggal").reset_index(drop=True).copy()
    if len(forecast_base_df) < WINDOW_SIZE:
        raise ValueError(f"Data historis tidak cukup untuk WINDOW_SIZE={WINDOW_SIZE}.")

    last_actual_value = float(forecast_base_df["jumlah_truk_asli"].iloc[-1])
    last_date = pd.to_datetime(forecast_base_df["tanggal"].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=int(horizon), freq="D")

    model = _load_keras_model(str(model_path))

    current_window = (
        forecast_base_df[TARGET_COL]
        .values[-WINDOW_SIZE:]
        .astype(float)
        .reshape(1, WINDOW_SIZE, 1)
    )

    current_actual = last_actual_value
    forecast_rows = []

    for step in range(1, int(horizon) + 1):
        pred_diff_scaled = model.predict(current_window, verbose=0)[0, 0]
        pred_diff = inverse_1d(scaler, np.array([pred_diff_scaled]))[0]
        predicted_actual = max(MIN_FORECAST_VALUE, current_actual + pred_diff)

        actual_diff_for_next_input = predicted_actual - current_actual
        actual_diff_scaled_for_next_input = scaler.transform(
            np.array([[actual_diff_for_next_input]])
        )[0, 0]

        forecast_rows.append(
            {
                "tanggal": future_dates[step - 1],
                "step": step,
                "forecast_jumlah_truk": predicted_actual,
                "predicted_diff": actual_diff_for_next_input,
                "predicted_diff_scaled": actual_diff_scaled_for_next_input,
                "model": model_name,
                "model_path": str(model_path),
            }
        )

        current_window = np.concatenate(
            [current_window[:, 1:, :], np.array([[[actual_diff_scaled_for_next_input]]])],
            axis=1,
        )
        current_actual = predicted_actual

    return pd.DataFrame(forecast_rows)


def render_forecast_page():
    st.title("Forecast")

    data = load_historical_daily()
    if data.error:
        st.error(data.error)
        return

    saved_result = st.session_state.get(FORECAST_STATE_KEY, {})
    saved_model_name = saved_result.get("model_name", list(MODEL_PATHS.keys())[0])
    saved_model_index = list(MODEL_PATHS.keys()).index(saved_model_name)

    with st.sidebar.form("forecast_form"):
        st.subheader("Pengaturan Forecast")
        model_name = st.selectbox(
            "Pilih Model",
            list(MODEL_PATHS.keys()),
            index=saved_model_index,
        )
        horizon = st.number_input(
            "Forecast Horizon (hari)",
            min_value=1,
            max_value=365,
            value=saved_result.get("horizon", 30),
            step=1,
        )
        history_days = st.number_input(
            "Jumlah histori yang ditampilkan (hari)",
            min_value=7,
            max_value=1000,
            value=saved_result.get("history_days", 90),
            step=1,
        )
        last_maintenance = st.date_input(
            "Tanggal maintenance terakhir",
            value=saved_result.get(
                "last_maintenance",
                data.daily["tanggal"].max().date(),
            ),
        )
        cycle_days = st.number_input(
            "Siklus maintenance (hari)",
            min_value=1,
            value=saved_result.get("cycle_days", 30),
            step=1,
        )
        buffer_days = st.number_input(
            "Buffer maintenance (hari)",
            min_value=0,
            value=saved_result.get("buffer_days", 5),
            step=1,
        )
        submitted = st.form_submit_button("Jalankan Forecast")

    if submitted:
        model_path = MODEL_PATHS[model_name]
        if not model_path.exists():
            st.error(model_not_found_message(model_name, model_path))
            return

        try:
            forecast_df = recursive_forecast(model_name, data.modeling, data.scaler, horizon)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Forecast gagal dijalankan: {exc}")
            return

        st.session_state[FORECAST_STATE_KEY] = {
            "forecast_df": forecast_df,
            "model_name": model_name,
            "horizon": int(horizon),
            "history_days": int(history_days),
            "last_maintenance": last_maintenance,
            "cycle_days": int(cycle_days),
            "buffer_days": int(buffer_days),
        }

    if FORECAST_STATE_KEY not in st.session_state:
        st.info("Atur parameter forecast pada sidebar, lalu klik Jalankan Forecast.")
        return

    saved_result = st.session_state[FORECAST_STATE_KEY]
    forecast_df = saved_result["forecast_df"]
    model_name = saved_result["model_name"]
    horizon = saved_result["horizon"]
    history_days = saved_result["history_days"]
    last_maintenance = saved_result["last_maintenance"]
    cycle_days = saved_result["cycle_days"]
    buffer_days = saved_result["buffer_days"]

    history_df = data.modeling.tail(int(history_days)).copy()
    recommendation = recommend_maintenance(
        forecast_df,
        last_maintenance,
        cycle_days,
        buffer_days,
    )

    table_df = forecast_df[["tanggal", "forecast_jumlah_truk"]].copy()
    table_df["forecast_jumlah_truk"] = table_df["forecast_jumlah_truk"].round(2)

    summary_panel, chart_panel = st.columns([0.28, 0.72], gap="small")

    with summary_panel:
        st.subheader("Ringkasan Forecast")
        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
        st.metric("Model", model_name)
        st.metric("Horizon", f"{int(horizon)} hari")
        st.metric(
            "Awal forecast",
            forecast_df["tanggal"].min().strftime("%Y-%m-%d"),
        )
        st.metric(
            "Akhir forecast",
            forecast_df["tanggal"].max().strftime("%Y-%m-%d"),
        )

    with chart_panel:
        st.subheader("Grafik Historis dan Forecast")
        st.plotly_chart(
            forecast_chart(
                history_df,
                forecast_df,
                model_name,
                recommendation_date=recommendation["tanggal_rekomendasi"],
                recommendation_value=recommendation["volume_rekomendasi"],
                buffer_start=recommendation["rentang_mulai"],
                buffer_end=recommendation["rentang_selesai"],
                height=390,
            ),
            use_container_width=True,
        )

    maintenance_panel, table_panel = st.columns([0.5, 0.5], gap="large")

    with maintenance_panel:
        st.subheader("Rekomendasi Maintenance")
        if recommendation["warning"]:
            st.warning(recommendation["warning"])

        maintenance_row1_col1, maintenance_row1_col2 = st.columns(
            [1.35, 0.65],
            gap="small",
        )
        maintenance_row1_col1.metric(
            "Tanggal maintenance terakhir",
            format_date_with_day(recommendation["maintenance_terakhir"]),
        )
        maintenance_row1_col2.metric(
            "Siklus maintenance",
            f"{recommendation['siklus']} hari",
        )

        maintenance_row2_col1, maintenance_row2_col2 = st.columns(
            [1.35, 0.65],
            gap="small",
        )
        maintenance_row2_col1.metric(
            "Estimasi maintenance berikutnya",
            format_date_with_day(recommendation["maintenance_berikutnya"]),
        )
        maintenance_row2_col2.metric(
            "Buffer",
            f"{recommendation['buffer']} hari",
        )

        maintenance_row3_col1, maintenance_row3_col2 = st.columns(
            [1.35, 0.65],
            gap="small",
        )
        if recommendation["tanggal_rekomendasi"] is not None:
            maintenance_row3_col1.metric(
                "Tanggal rekomendasi maintenance",
                format_date_with_day(recommendation["tanggal_rekomendasi"]),
            )
            maintenance_row3_col2.metric(
                "Prediksi volume truk",
                f"{recommendation['volume_rekomendasi']:,.2f}",
            )
        else:
            maintenance_row3_col1.metric("Tanggal rekomendasi maintenance", "-")
            maintenance_row3_col2.metric("Prediksi volume truk", "-")

        st.write(
            "Rentang evaluasi: "
            f"{format_date_with_day(recommendation['rentang_mulai'])} sampai "
            f"{format_date_with_day(recommendation['rentang_selesai'])}"
        )

        if recommendation["tanggal_rekomendasi"] is not None:
            st.info(
                "Alasan rekomendasi: Memiliki prediksi volume arus truk terendah pada rentang evaluasi sehingga "
                "berpotensi meminimalkan gangguan operasional saat maintenance dilakukan."
            )

    with table_panel:
        st.subheader("Tabel Hasil Forecast")
        display_table_df = table_df.copy()
        display_table_df["tanggal"] = pd.to_datetime(display_table_df["tanggal"]).dt.strftime("%Y-%m-%d")
        st.dataframe(
            display_table_df,
            use_container_width=True,
            height=430,
            hide_index=True,
        )
        st.download_button(
            "Download CSV hasil forecast",
            data=table_df.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_{model_name.lower().replace(' ', '_').replace('-', '_')}.csv",
            mime="text/csv",
        )
