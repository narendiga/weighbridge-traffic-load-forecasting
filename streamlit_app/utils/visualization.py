import plotly.graph_objects as go


def historical_chart(daily, show_title=True, height=None, bottom_margin=20):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["tanggal"],
            y=daily["jumlah_truk"],
            mode="lines",
            name="Data aktual",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Jumlah Truk",
        hovermode="x unified",
        height=height,
        margin=dict(l=20, r=20, t=50 if show_title else 20, b=bottom_margin),
    )
    if show_title:
        fig.update_layout(title="Historis Volume Truk Harian")
    return fig


def forecast_chart(
    history_df,
    forecast_df,
    model_label,
    recommendation_date=None,
    recommendation_value=None,
    buffer_start=None,
    buffer_end=None,
    height=None,
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df["tanggal"],
            y=history_df["jumlah_truk_asli"],
            mode="lines",
            name="Data aktual",
            line=dict(color="#1f77b4", width=2),
        )
    )

    forecast_dates = forecast_df["tanggal"].tolist()
    forecast_values = forecast_df["forecast_jumlah_truk"].tolist()
    if not history_df.empty:
        forecast_dates = [history_df["tanggal"].iloc[-1], *forecast_dates]
        forecast_values = [history_df["jumlah_truk_asli"].iloc[-1], *forecast_values]

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode="lines+markers",
            name="Data forecast",
            line=dict(color="#d62728", width=2, dash="dash"),
        )
    )

    if buffer_start is not None and buffer_end is not None:
        fig.add_vrect(
            x0=buffer_start,
            x1=buffer_end,
            fillcolor="LightSalmon",
            opacity=0.2,
            layer="below",
            line_width=0,
            annotation_text="Rentang Buffer",
            annotation_position="top left",
        )

    if recommendation_date is not None and recommendation_value is not None:
        fig.add_trace(
            go.Scatter(
                x=[recommendation_date],
                y=[recommendation_value],
                mode="markers",
                name="Rekomendasi Maintenance",
                marker=dict(
                    symbol="star",
                    size=18,
                    color="#ffbf00",
                    line=dict(color="#7f6000", width=2),
                ),
                hovertemplate=(
                    "Rekomendasi Maintenance"
                    "<br>Tanggal: %{x|%Y-%m-%d}"
                    "<br>Forecast jumlah truk: %{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Jumlah Truk",
        hovermode="x unified",
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def aggregation_chart(df, x_col, y_col, title, show_title=True, height=None):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df[x_col],
            y=df[y_col],
            name=title,
            marker_color="#2ca02c",
        )
    )
    fig.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Jumlah Truk",
        hovermode="x unified",
        height=height,
        margin=dict(l=20, r=20, t=50 if show_title else 20, b=20),
    )
    if show_title:
        fig.update_layout(title=title)
    return fig


def pattern_bar_chart(pattern_df, busiest_label, quietest_label, y_axis_title):
    colors = [
        "#c2413b" if label == busiest_label else "#19765a" if label == quietest_label else "#2878b8"
        for label in pattern_df["label"]
    ]

    fig = go.Figure(
        go.Bar(
            x=pattern_df["label"],
            y=pattern_df["nilai"],
            marker_color=colors,
            hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
        )
    )
    label_count = len(pattern_df)
    tick_angle = -45 if label_count > 12 else -25 if label_count > 7 else 0

    fig.update_layout(
        xaxis_title=None,
        yaxis_title=y_axis_title,
        hovermode="x",
        height=215,
        margin=dict(l=20, r=20, t=5, b=0),
        xaxis=dict(
            tickangle=tick_angle,
            tickfont=dict(size=9),
            automargin=True,
        ),
    )
    return fig
