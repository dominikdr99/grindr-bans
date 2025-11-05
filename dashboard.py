from typing import Iterable

import pandas as pd
import streamlit as st
import altair as alt
import requests

from monitor import SOURCES, fetch_records, normalize_row
API_ERROR_HELP = "Verify Streamlit secrets or environment variables for NocoDB access and redeploy."


def fetch_all_sources(sources: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []
    missing_sources = True
    for source in sources:
        table_id = source.get("table_id")
        if not table_id:
            continue
        missing_sources = False
        label = source.get("label") or "Unknown"
        view_id = source.get("view_id")
        try:
            records = fetch_records(table_id=table_id, view_id=view_id)
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to pull data for {label}: {exc}") from exc
        rows.extend(normalize_row(record, talent=label) for record in records)

    if missing_sources:
        raise RuntimeError("No NocoDB sources configured.")

    return rows


@st.cache_data(ttl=1800)
def load_bans() -> pd.DataFrame:
    if not SOURCES:
        raise RuntimeError("No NocoDB sources configured.")

    df = pd.DataFrame(fetch_all_sources(SOURCES))
    if df.empty:
        return df
    for col in ("banned_at_local", "banned_at_utc"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    if df["banned_at_local"].dt.tz is not None:
        df["banned_at_local"] = df["banned_at_local"].dt.tz_convert(None)
    df["talent"] = df["talent"].fillna("Unknown")
    df["banned_date"] = df["banned_at_local"].dt.date
    return df


def sidebar_filters(df: pd.DataFrame):
    if df.empty:
        st.sidebar.info("No bans found yet.")
        return df, []

    st.sidebar.header("Filters")

    talents = sorted(df["talent"].dropna().unique())
    reasons = sorted(df["reason"].dropna().unique())
    platforms = sorted(df["platform"].dropna().unique())
    models = sorted(df["model"].dropna().unique())

    talent_filter = st.sidebar.multiselect("Model", talents, default=talents)
    reason_filter = st.sidebar.multiselect("Ban reason", reasons)
    platform_filter = st.sidebar.multiselect("Device manufacturer", platforms)
    model_filter = st.sidebar.multiselect("Device model", models)

    has_timestamps = df["banned_at_local"].notna().any()
    min_date = df["banned_date"].dropna().min()
    max_date = df["banned_date"].dropna().max()

    range_options = [
        "All time",
        "Last 24 hours",
        "Last 7 days",
        "Last 30 days",
        "Last N days",
        "Custom range",
    ]
    range_choice = st.sidebar.selectbox("Time range", range_options, index=0)
    custom_days = None
    custom_range = ()
    if range_choice == "Last N days":
        custom_days = st.sidebar.number_input("Number of days", min_value=1, value=14, step=1)
    elif range_choice == "Custom range" and not (pd.isna(min_date) or pd.isna(max_date)):
        custom_range = st.sidebar.date_input(
            "Ban date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    mask = pd.Series(True, index=df.index)
    selected_talents = talent_filter or talents
    if selected_talents:
        mask &= df["talent"].isin(selected_talents)
    if reason_filter:
        mask &= df["reason"].isin(reason_filter)
    if platform_filter:
        mask &= df["platform"].isin(platform_filter)
    if model_filter:
        mask &= df["model"].isin(model_filter)

    if has_timestamps:
        now = pd.Timestamp.utcnow().tz_localize(None)
        data_max = df["banned_at_local"].max()
        if pd.notna(data_max) and data_max > now:
            now = data_max
        if range_choice == "Last 24 hours":
            cutoff = now - pd.Timedelta(hours=24)
            mask &= df["banned_at_local"] >= cutoff
        elif range_choice == "Last 7 days":
            cutoff = now - pd.Timedelta(days=7)
            mask &= df["banned_at_local"] >= cutoff
        elif range_choice == "Last 30 days":
            cutoff = now - pd.Timedelta(days=30)
            mask &= df["banned_at_local"] >= cutoff
        elif range_choice == "Last N days" and custom_days:
            cutoff = now - pd.Timedelta(days=custom_days)
            mask &= df["banned_at_local"] >= cutoff
        elif range_choice == "Custom range" and isinstance(custom_range, tuple) and len(custom_range) == 2:
            start, end = custom_range
            if start:
                mask &= df["banned_date"] >= pd.to_datetime(start).date()
            if end:
                mask &= df["banned_date"] <= pd.to_datetime(end).date()

    return df.loc[mask].copy(), selected_talents


def summary_metrics(df: pd.DataFrame, selected_talents):
    total_bans = len(df)
    now = pd.Timestamp.utcnow().tz_localize(None)
    df_times = df.dropna(subset=["banned_at_local"]).copy()
    if not df_times.empty and df_times["banned_at_local"].max() > now:
        now = df_times["banned_at_local"].max()

    today = now.date()
    window_12h = now - pd.Timedelta(hours=12)
    window_4h = now - pd.Timedelta(hours=4)
    window_1h = now - pd.Timedelta(hours=1)

    todays_bans = df_times[df_times["banned_at_local"].dt.date == today].shape[0]
    last_12h = df_times[df_times["banned_at_local"] >= window_12h].shape[0]
    last_4h = df_times[df_times["banned_at_local"] >= window_4h].shape[0]
    last_1h = df_times[df_times["banned_at_local"] >= window_1h].shape[0]

    cols = st.columns(5)
    cols[0].metric("Total bans", f"{total_bans:,}")
    cols[1].metric("Today", f"{todays_bans:,}")
    cols[2].metric("Last 12h", f"{last_12h:,}")
    cols[3].metric("Last 4h", f"{last_4h:,}")
    cols[4].metric("Last hour", f"{last_1h:,}")

    if df.empty or not selected_talents:
        return

    breakdown = (
        df.groupby("talent")
        .agg(bans=("account_id", "count"))
        .reset_index()
        .sort_values("bans", ascending=False)
    )

    if df_times.empty:
        today_counts = pd.Series(dtype="int64")
        last_12h_counts = pd.Series(dtype="int64")
    else:
        today_counts = (
            df_times[df_times["banned_at_local"].dt.date == today]
            .groupby("talent")
            .size()
        )
        last_12h_counts = (
            df_times[df_times["banned_at_local"] >= window_12h]
            .groupby("talent")
            .size()
        )

    breakdown["Today"] = breakdown["talent"].map(today_counts).fillna(0).astype(int)
    breakdown["Last 12h"] = breakdown["talent"].map(last_12h_counts).fillna(0).astype(int)
    breakdown["bans"] = breakdown["bans"].astype(int)
    breakdown.rename(
        columns={
            "talent": "Model",
            "bans": "Bans",
        },
        inplace=True,
    )
    st.caption("Breakdown by model and recent activity")
    st.dataframe(breakdown, use_container_width=True, hide_index=True)


def time_series_chart(df: pd.DataFrame, granularity: str):
    if df.empty or df["banned_at_local"].isna().all():
        return

    freq = "D" if granularity == "Daily" else "H"
    daily = (
        df.dropna(subset=["banned_at_local"])
        .groupby([pd.Grouper(key="banned_at_local", freq=freq), "talent"])
        .size()
        .reset_index(name="count")
    )
    daily = daily.rename(columns={"banned_at_local": "period"})
    chart = (
        alt.Chart(daily)
        .mark_line(point=True)
        .encode(
            x=alt.X("period:T", title="Date & time" if granularity == "Hourly" else "Date"),
            y=alt.Y("count:Q", title=f"Bans per {granularity.lower()}"),
            color=alt.Color("talent:N", title="Model"),
            tooltip=["period:T", "talent:N", "count:Q"],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def distribution_charts(df: pd.DataFrame):
    if df.empty:
        return

    st.subheader("Ban Distribution")
    chart_columns = st.columns(3)
    chart_specs = [
        ("model", "Device Model"),
        ("platform", "Device Manufacturer"),
        ("talent", "Model"),
    ]

    for idx, (column, label) in enumerate(chart_specs):
        if column not in df.columns:
            chart_columns[idx].info(f"No data for {label.lower()}.")
            continue
        data = (
            df[column]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
            .rename(columns={"index": "category", column: "Bans"})
        )
        if data.empty or data["Bans"].sum() == 0:
            chart_columns[idx].info(f"No data for {label.lower()}.")
            continue
        data["Share"] = data["Bans"] / data["Bans"].sum()
        chart = (
            alt.Chart(data)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta("Bans:Q", stack=True),
                color=alt.Color("category:N", title=label),
                tooltip=[
                    alt.Tooltip("category:N", title=label),
                    alt.Tooltip("Bans:Q", title="Bans"),
                    alt.Tooltip("Share:Q", title="Share", format=".1%"),
                ],
            )
            .properties(title=label)
        )
        chart_columns[idx].altair_chart(chart, use_container_width=True)


def bans_table(df: pd.DataFrame):
    display_df = df[
        [
            "banned_at_local",
            "talent",
            "account_id",
            "username",
            "email",
            "model",
            "platform",
            "reason",
            "about_me",
        ]
    ].rename(
        columns={
            "banned_at_local": "Banned At",
            "talent": "Model",
            "account_id": "Account ID",
            "username": "Display Name",
            "model": "Device Model",
            "platform": "Device Manufacturer",
            "reason": "Ban Reason",
            "about_me": "About Me",
        }
    )
    st.dataframe(display_df, use_container_width=True)


def main():
    st.set_page_config(
        page_title="Ban Monitor",
        page_icon="🚫",
        layout="wide",
    )
    st.title("🚫 Ban Monitor Dashboard")
    st.caption("Explore bans pulled directly from NocoDB.")

    try:
        df = load_bans()
    except RuntimeError as exc:
        st.error(f"{exc}\n{API_ERROR_HELP}")
        st.stop()
    if df.empty:
        st.info("No bans available yet. Confirm your NocoDB tables contain data.")
        return

    filtered_df, selected_talents = sidebar_filters(df)

    summary_metrics(filtered_df, selected_talents)
    distribution_charts(filtered_df)
    st.subheader("Trends")
    granularity = st.radio("Time granularity", ["Daily", "Hourly"], horizontal=True)
    time_series_chart(filtered_df, granularity)

    st.subheader("Ban Details")
    bans_table(filtered_df)


if __name__ == "__main__":
    main()
