import pandas as pd


WARNING_OUTSIDE_FORECAST = (
    "Rentang evaluasi maintenance berada di luar periode forecast. "
    "Perbesar horizon forecast atau ubah tanggal maintenance terakhir/siklus maintenance."
)


def recommend_maintenance(forecast_df, last_maintenance_date, cycle_days, buffer_days):
    maintenance_date = pd.Timestamp(last_maintenance_date).normalize()
    cycle_days = int(cycle_days)
    buffer_days = int(buffer_days)

    next_maintenance_date = maintenance_date + pd.Timedelta(days=cycle_days)
    start_date = next_maintenance_date - pd.Timedelta(days=buffer_days)
    end_date = next_maintenance_date + pd.Timedelta(days=buffer_days)

    forecast_dates = pd.to_datetime(forecast_df["tanggal"])
    mask = (forecast_dates >= start_date) & (forecast_dates <= end_date)
    candidates = forecast_df.loc[mask].copy()

    result = {
        "maintenance_terakhir": maintenance_date,
        "siklus": cycle_days,
        "maintenance_berikutnya": next_maintenance_date,
        "buffer": buffer_days,
        "rentang_mulai": start_date,
        "rentang_selesai": end_date,
        "warning": None,
        "tanggal_rekomendasi": None,
        "volume_rekomendasi": None,
    }

    if candidates.empty:
        result["warning"] = WARNING_OUTSIDE_FORECAST
        return result

    best = candidates.loc[candidates["forecast_jumlah_truk"].idxmin()]
    result["tanggal_rekomendasi"] = pd.Timestamp(best["tanggal"])
    result["volume_rekomendasi"] = float(best["forecast_jumlah_truk"])
    return result
