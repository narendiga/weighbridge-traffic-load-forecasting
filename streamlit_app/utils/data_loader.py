from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "weighbridge_fix.xlsx"
DATE_COLUMN = "waktu_closing"
TARGET_COLUMN = "jumlah_truk"
INDONESIAN_DAY_NAMES = {
    0: "Senin",
    1: "Selasa",
    2: "Rabu",
    3: "Kamis",
    4: "Jumat",
    5: "Sabtu",
    6: "Minggu",
}
INDONESIAN_MONTH_NAMES = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember",
}


@dataclass
class HistoricalData:
    raw: pd.DataFrame | None
    daily: pd.DataFrame | None
    modeling: pd.DataFrame | None
    scaler: object | None
    error: str | None = None


class SimpleMinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.data_min_ = np.nanmin(values, axis=0)
        self.data_max_ = np.nanmax(values, axis=0)
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=float)
        low, high = self.feature_range
        denom = np.where((self.data_max_ - self.data_min_) == 0, 1, self.data_max_ - self.data_min_)
        scaled = (values - self.data_min_) / denom
        return scaled * (high - low) + low

    def fit_transform(self, values):
        return self.fit(values).transform(values)

    def inverse_transform(self, values):
        values = np.asarray(values, dtype=float)
        low, high = self.feature_range
        denom = np.where((high - low) == 0, 1, high - low)
        unscaled = (values - low) / denom
        return unscaled * (self.data_max_ - self.data_min_) + self.data_min_


def _local_poly_estimate(series, target_date, order, window_days):
    target_date = pd.Timestamp(target_date)
    start = target_date - pd.Timedelta(days=window_days)
    end = target_date + pd.Timedelta(days=window_days)
    segment = series.loc[start:end].copy()
    valid = segment.dropna()

    if len(valid) < order + 2:
        return np.nan

    x = np.array([(d - target_date).days for d in valid.index], dtype=float)
    y = valid.values.astype(float)

    if len(np.unique(x)) <= order:
        return np.nan

    try:
        coeffs = np.polyfit(x, y, deg=order)
        return float(np.poly1d(coeffs)(0))
    except Exception:
        return np.nan


def _apply_notebook_interpolation(daily):
    selected_interp_order = 2
    daily_before_interp = daily[TARGET_COLUMN].copy()
    series = daily_before_interp.copy().astype(float)
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()

    interp_global = series.interpolate(
        method="polynomial",
        order=selected_interp_order,
        limit_direction="both",
    )

    target_dates = pd.to_datetime(["2024-04-11", "2024-04-12"])
    for target_date in target_dates:
        if target_date not in interp_global.index or pd.notna(series.loc[target_date]):
            continue

        best_val = np.nan
        for window_days in [3, 5, 7, 9, 11, 13, 15]:
            estimate = _local_poly_estimate(series, target_date, order=selected_interp_order, window_days=window_days)
            if pd.notna(estimate) and estimate >= 0:
                best_val = estimate
                break

        if pd.notna(best_val):
            interp_global.loc[target_date] = best_val

    daily[TARGET_COLUMN] = interp_global.round().clip(lower=0).astype(int)
    return daily


def summarize_historical_patterns(raw, daily):
    timestamps = pd.to_datetime(raw[DATE_COLUMN], errors="coerce").dropna()
    raw_hourly = pd.DataFrame(
        {
            "tanggal": timestamps.dt.normalize(),
            "jam": timestamps.dt.hour,
        }
    )

    full_dates = pd.date_range(daily["tanggal"].min(), daily["tanggal"].max(), freq="D")
    full_hours = pd.MultiIndex.from_product(
        [full_dates, range(24)],
        names=["tanggal", "jam"],
    )
    hourly_counts = raw_hourly.groupby(["tanggal", "jam"]).size().reindex(full_hours, fill_value=0)
    hourly_average = hourly_counts.groupby("jam").mean()

    daily_values = daily[["tanggal", TARGET_COLUMN]].copy()
    daily_values["hari_ke"] = daily_values["tanggal"].dt.dayofweek
    daily_values["bulan_ke"] = daily_values["tanggal"].dt.month

    day_average = daily_values.groupby("hari_ke")[TARGET_COLUMN].mean()
    month_average = daily_values.groupby("bulan_ke")[TARGET_COLUMN].mean()

    busiest_hour = int(hourly_average.idxmax())
    quietest_hour = int(hourly_average.idxmin())
    busiest_day = int(day_average.idxmax())
    quietest_day = int(day_average.idxmin())
    busiest_month = int(month_average.idxmax())
    quietest_month = int(month_average.idxmin())

    return {
        "jam": {
            "tersibuk": f"{busiest_hour:02d}:00",
            "nilai_tersibuk": float(hourly_average.loc[busiest_hour]),
            "tersepi": f"{quietest_hour:02d}:00",
            "nilai_tersepi": float(hourly_average.loc[quietest_hour]),
            "satuan": "rata-rata transaksi/jam",
        },
        "hari": {
            "tersibuk": INDONESIAN_DAY_NAMES[busiest_day],
            "nilai_tersibuk": float(day_average.loc[busiest_day]),
            "tersepi": INDONESIAN_DAY_NAMES[quietest_day],
            "nilai_tersepi": float(day_average.loc[quietest_day]),
            "satuan": "rata-rata truk/hari",
        },
        "bulan": {
            "tersibuk": INDONESIAN_MONTH_NAMES[busiest_month],
            "nilai_tersibuk": float(month_average.loc[busiest_month]),
            "tersepi": INDONESIAN_MONTH_NAMES[quietest_month],
            "nilai_tersepi": float(month_average.loc[quietest_month]),
            "satuan": "rata-rata truk/hari",
        },
    }


def build_historical_pattern_series(raw, daily):
    timestamps = pd.to_datetime(raw[DATE_COLUMN], errors="coerce").dropna()
    raw_hourly = pd.DataFrame(
        {
            "tanggal": timestamps.dt.normalize(),
            "jam": timestamps.dt.hour,
        }
    )

    full_dates = pd.date_range(daily["tanggal"].min(), daily["tanggal"].max(), freq="D")
    full_hours = pd.MultiIndex.from_product(
        [full_dates, range(24)],
        names=["tanggal", "jam"],
    )
    hourly_counts = raw_hourly.groupby(["tanggal", "jam"]).size().reindex(full_hours, fill_value=0)
    hourly_average = hourly_counts.groupby("jam").mean()

    daily_values = daily[["tanggal", TARGET_COLUMN]].copy()
    daily_values["hari_ke"] = daily_values["tanggal"].dt.dayofweek
    daily_values["bulan_ke"] = daily_values["tanggal"].dt.month

    day_average = daily_values.groupby("hari_ke")[TARGET_COLUMN].mean().reindex(range(7))
    month_average = daily_values.groupby("bulan_ke")[TARGET_COLUMN].mean().reindex(range(1, 13))

    return {
        "jam": pd.DataFrame(
            {
                "label": [f"{hour:02d}:00" for hour in hourly_average.index],
                "nilai": hourly_average.values,
            }
        ),
        "hari": pd.DataFrame(
            {
                "label": [INDONESIAN_DAY_NAMES[day] for day in day_average.index],
                "nilai": day_average.values,
            }
        ),
        "bulan": pd.DataFrame(
            {
                "label": [INDONESIAN_MONTH_NAMES[month] for month in month_average.index],
                "nilai": month_average.values,
            }
        ),
    }


@st.cache_data(show_spinner=False)
def load_historical_daily():
    if not DATA_PATH.exists():
        return HistoricalData(
            raw=None,
            daily=None,
            modeling=None,
            scaler=None,
            error=f"File historis tidak ditemukan: {DATA_PATH}",
        )

    try:
        raw = pd.read_excel(DATA_PATH)
    except Exception as exc:
        return HistoricalData(raw=None, daily=None, modeling=None, scaler=None, error=f"Gagal membaca file historis: {exc}")

    if DATE_COLUMN not in raw.columns:
        return HistoricalData(
            raw=raw,
            daily=None,
            modeling=None,
            scaler=None,
            error=f"Kolom tanggal '{DATE_COLUMN}' tidak ditemukan pada file historis.",
        )

    try:
        raw = raw.copy()
        raw["closing_date"] = pd.to_datetime(raw[DATE_COLUMN], errors="coerce").dt.normalize()
        raw = raw.dropna(subset=["closing_date"])

        daily = raw.groupby("closing_date").size().reset_index(name=TARGET_COLUMN)
        daily = daily.set_index("closing_date")

        full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(full_range)
        daily.index.name = "tanggal"
        daily = _apply_notebook_interpolation(daily)
        daily = daily.reset_index().copy()
        daily["tanggal"] = pd.to_datetime(daily["tanggal"])
        daily["day_of_week"] = daily["tanggal"].dt.day_name()
        daily["tanggal_ke"] = daily["tanggal"].dt.day
        daily["bulan_ke"] = daily["tanggal"].dt.month

        modeling = daily.sort_values("tanggal").reset_index(drop=True).copy()
        modeling["jumlah_truk_asli"] = modeling[TARGET_COLUMN].astype(float)
        modeling["jumlah_truk_diff"] = modeling["jumlah_truk_asli"].diff()
        modeling = modeling.dropna(subset=["jumlah_truk_diff"]).reset_index(drop=True)

        scaler = SimpleMinMaxScaler(feature_range=(0, 1))
        modeling["jumlah_truk_diff_scaled"] = scaler.fit_transform(
            modeling[["jumlah_truk_diff"]].values
        ).flatten()
    except Exception as exc:
        return HistoricalData(raw=raw, daily=None, modeling=None, scaler=None, error=f"Gagal menjalankan preprocessing notebook: {exc}")

    if scaler.data_min_ is None or scaler.data_max_ is None:
        return HistoricalData(raw=raw, daily=daily, modeling=modeling, scaler=None, error="Scaler tidak berhasil dibuat dari data historis.")

    return HistoricalData(raw=raw, daily=daily, modeling=modeling, scaler=scaler)
