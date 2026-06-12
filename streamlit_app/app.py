from pathlib import Path

import streamlit as st

from utils.data_loader import (
    build_historical_pattern_series,
    load_historical_daily,
    summarize_historical_patterns,
)
from utils.forecasting import MODEL_PATHS, render_forecast_page
from utils.visualization import aggregation_chart, historical_chart, pattern_bar_chart


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "weighbridge_fix.xlsx"
MODEL_DIR = PROJECT_ROOT / "models"


st.set_page_config(
    page_title="Forecasting Volume Truk PT XYZ",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] {
        padding-top: 1rem !important;
        padding-bottom: 1.25rem !important;
    }

    [data-testid="stSidebarUserContent"] {
        padding-top: 1.5rem !important;
        margin-top: -3.75rem !important;
    }

    section[data-testid="stSidebar"] {
        width: 280px !important;
        min-width: 280px !important;
    }

    section[data-testid="stSidebar"] > div {
        width: 280px !important;
    }

    .dashboard-kpi {
        min-height: 125px;
        padding: 0.85rem 0.8rem;
        border: 1px solid #e3e8ef;
        border-top: 3px solid #2878b8;
        border-radius: 6px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .dashboard-kpi-label {
        color: #172033;
        font-size: 0.9rem;
        font-weight: 650;
        line-height: 1.2;
    }

    .dashboard-kpi-value {
        color: #2878b8;
        font-size: 1.45rem;
        font-weight: 650;
        line-height: 1.2;
    }

    .dashboard-kpi-note {
        color: #596579;
        font-size: 0.78rem;
        line-height: 1.25;
    }

    .pattern-card {
        min-height: 205px;
        padding: 0.7rem 0.8rem;
        border: 1px solid #e3e8ef;
        border-radius: 6px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
    }

    .pattern-title {
        color: #172033;
        font-size: 0.9rem;
        font-weight: 650;
        margin-bottom: 0.5rem;
    }

    .pattern-grid {
        display: flex;
        flex-direction: column;
        gap: 0.65rem;
    }

    .pattern-label {
        color: #7b8798;
        font-size: 0.72rem;
        margin-bottom: 0.1rem;
    }

    .pattern-busy, .pattern-quiet {
        font-size: 1.25rem;
        font-weight: 650;
        line-height: 1.2;
    }

    .pattern-busy {
        color: #c2413b;
    }

    .pattern-quiet {
        color: #19765a;
    }

    .pattern-value {
        color: #596579;
        font-size: 0.76rem;
        line-height: 1.25;
        margin-top: 0.25rem;
    }

    .pattern-grid > div + div {
        margin-top: 14px;
    }

    .dashboard-section-title {
        color: #172033;
        font-size: 1.05rem;
        font-weight: 650;
        line-height: 1.2;
        margin: 0.35rem 0 0 1rem;
    }

    .dashboard-pattern-title {
        color: #172033;
        font-size: 1.05rem;
        font-weight: 650;
        line-height: 1.2;
        margin: -0.65rem 0 0.2rem 1rem;
    }

    .dashboard-info {
        color: #0865b5;
        background: #e6f2ff;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin-top: -0.4rem;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
        line-height: 1.35;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


def apply_forecast_font_styles():
    st.markdown(
        """
        <style>
        /* Judul halaman Forecast */
        h1 {
            font-size: 2.5rem !important;
        }

        /* Judul bagian Ringkasan, Grafik, Rekomendasi, dan Tabel */
        h3 {
            font-size: 1.4rem !important;
        }

        /* Nilai utama model, horizon, tanggal, dan prediksi */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
        }

        /* Label Model, Horizon, Siklus, Buffer, dan lainnya */
        [data-testid="stMetricLabel"] {
            font-size: 1.5rem !important;
        }

        /* Label input pada sidebar */
        section[data-testid="stSidebar"] label {
            font-size: 0.85rem !important;
        }

        /* Judul Pengaturan Forecast pada sidebar */
        section[data-testid="stSidebar"] h3 {
            font-size: 1.1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard():
    st.title("Dashboard Volume Truk")

    data = load_historical_daily()
    if data.error:
        st.error(data.error)
        return

    daily = data.daily
    patterns = summarize_historical_patterns(data.raw, daily)
    pattern_series = build_historical_pattern_series(data.raw, daily)

    kpi_values = [
        ("Jumlah Data Historis", f"{len(daily):,}", "hari dalam deret waktu"),
        ("Rata-rata Volume", f"{daily['jumlah_truk'].mean():,.2f}", "truk per hari"),
        ("Total Transaksi", f"{len(data.raw):,}", "transaksi historis"),
        (
            "Periode Data",
            f"{daily['tanggal'].min():%Y-%m-%d}",
            f"sampai {daily['tanggal'].max():%Y-%m-%d}",
        ),
    ]

    kpi_left_panel, kpi_middle_panel, history_chart_panel = st.columns(
        [0.155, 0.155, 0.71],
        gap="small",
    )

    with kpi_left_panel:
        for label, value, note in kpi_values[:2]:
            st.markdown(
                f"""
                <div class="dashboard-kpi">
                    <div class="dashboard-kpi-label">{label}</div>
                    <div class="dashboard-kpi-value">{value}</div>
                    <div class="dashboard-kpi-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height: 0.25rem;'></div>", unsafe_allow_html=True)

    with kpi_middle_panel:
        for label, value, note in kpi_values[2:]:
            st.markdown(
                f"""
                <div class="dashboard-kpi">
                    <div class="dashboard-kpi-label">{label}</div>
                    <div class="dashboard-kpi-value">{value}</div>
                    <div class="dashboard-kpi-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height: 0.25rem;'></div>", unsafe_allow_html=True)

    with history_chart_panel:
        st.markdown(
            '<div class="dashboard-section-title">Volume Truk Harian</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            historical_chart(daily, show_title=False, height=270, bottom_margin=5),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown(
        '<div class="dashboard-pattern-title">Kesimpulan Pola Historis</div>',
        unsafe_allow_html=True,
    )
    pattern_titles = {
        "jam": "Pola Jam Operasional",
        "hari": "Pola Hari dalam Minggu",
        "bulan": "Pola Bulan dalam Tahun",
    }

    (
        pattern_chart_panel,
        pattern_hour_panel,
        pattern_day_panel,
        pattern_month_panel,
    ) = st.columns([0.49, 0.17, 0.17, 0.17], gap="small")

    pattern_options = {
        "Jam Operasional": ("jam", "Rata-rata Transaksi/Jam"),
        "Hari dalam Minggu": ("hari", "Rata-rata Truk/Hari"),
        "Bulan dalam Tahun": ("bulan", "Rata-rata Truk/Hari"),
    }

    with pattern_chart_panel:
        selected_pattern_label = st.session_state.get(
            "dashboard_pattern_selector",
            "Jam Operasional",
        )
        selected_pattern_key, selected_y_title = pattern_options[selected_pattern_label]
        st.plotly_chart(
            pattern_bar_chart(
                pattern_series[selected_pattern_key],
                patterns[selected_pattern_key]["tersibuk"],
                patterns[selected_pattern_key]["tersepi"],
                selected_y_title,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.radio(
            "Pilih pola",
            list(pattern_options.keys()),
            horizontal=True,
            label_visibility="collapsed",
            key="dashboard_pattern_selector",
        )

    pattern_card_panels = [
        (pattern_hour_panel, "jam"),
        (pattern_day_panel, "hari"),
        (pattern_month_panel, "bulan"),
    ]

    for panel, pattern_key in pattern_card_panels:
        with panel:
            pattern = patterns[pattern_key]
            st.markdown(
                f"""
                <div class="pattern-card">
                    <div class="pattern-title">{pattern_titles[pattern_key]}</div>
                    <div class="pattern-grid">
                        <div>
                            <div class="pattern-label">Tersibuk</div>
                            <div class="pattern-busy">{pattern['tersibuk']}</div>
                            <div class="pattern-value">{pattern['nilai_tersibuk']:.2f} {pattern['satuan']}</div>
                        </div>
                        <div>
                            <div class="pattern-label">Tersepi</div>
                            <div class="pattern-quiet">{pattern['tersepi']}</div>
                            <div class="pattern-value">{pattern['nilai_tersepi']:.2f} {pattern['satuan']}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="dashboard-info">
            Pola historis digunakan sebagai informasi pendukung. Rekomendasi maintenance
            tetap dipilih dari hasil forecast volume truk terendah pada rentang evaluasi.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detail_data():
    st.title("Detail Data")

    data = load_historical_daily()
    if data.error:
        st.error(data.error)
        return

    daily = data.daily.copy()
    tab_harian, tab_mingguan, tab_bulanan = st.tabs(["Harian", "Mingguan", "Bulanan"])

    with tab_harian:
        table_panel, chart_panel = st.columns([0.42, 0.58], gap="medium")

        with table_panel:
            st.subheader("Tabel Data Harian")
            daily_table = daily.copy()
            daily_table["tanggal"] = daily_table["tanggal"].dt.strftime("%Y-%m-%d")
            st.dataframe(
                daily_table,
                use_container_width=True,
                height=390,
                hide_index=True,
            )

        with chart_panel:
            st.subheader("Grafik Data Harian")
            st.plotly_chart(
                historical_chart(daily, show_title=False, height=390),
                use_container_width=True,
            )

    with tab_mingguan:
        weekly = (
            daily.set_index("tanggal")["jumlah_truk"]
            .resample("W")
            .agg(sum_jumlah_truk="sum", mean_jumlah_truk="mean")
            .reset_index()
        )
        table_panel, chart_panel = st.columns([0.42, 0.58], gap="medium")

        with table_panel:
            st.subheader("Tabel Agregasi Mingguan")
            weekly_table = weekly.copy()
            weekly_table["tanggal"] = weekly_table["tanggal"].dt.strftime("%Y-%m-%d")
            st.dataframe(
                weekly_table,
                use_container_width=True,
                height=390,
                hide_index=True,
            )

        with chart_panel:
            st.subheader("Grafik Total Truk Mingguan")
            st.plotly_chart(
                aggregation_chart(
                    weekly,
                    "tanggal",
                    "sum_jumlah_truk",
                    "Total Truk Mingguan",
                    show_title=False,
                    height=390,
                ),
                use_container_width=True,
            )

    with tab_bulanan:
        monthly = (
            daily.set_index("tanggal")["jumlah_truk"]
            .resample("ME")
            .agg(sum_jumlah_truk="sum", mean_jumlah_truk="mean")
            .reset_index()
        )
        table_panel, chart_panel = st.columns([0.42, 0.58], gap="medium")

        with table_panel:
            st.subheader("Tabel Agregasi Bulanan")
            monthly_table = monthly.copy()
            monthly_table["tanggal"] = monthly_table["tanggal"].dt.strftime("%Y-%m")
            st.dataframe(
                monthly_table,
                use_container_width=True,
                height=390,
                hide_index=True,
            )

        with chart_panel:
            st.subheader("Grafik Total Truk Bulanan")
            st.plotly_chart(
                aggregation_chart(
                    monthly,
                    "tanggal",
                    "sum_jumlah_truk",
                    "Total Truk Bulanan",
                    show_title=False,
                    height=390,
                ),
                use_container_width=True,
            )


def main():
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Menu", ["Dashboard", "Forecast", "Detail Data"])

    if page == "Dashboard":
        render_dashboard()
    elif page == "Forecast":
        apply_forecast_font_styles()
        render_forecast_page()
    else:
        render_detail_data()


if __name__ == "__main__":
    main()
