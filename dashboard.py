from typing import Iterable

import pandas as pd
import streamlit as st
import altair as alt
import requests

from monitor import SOURCES, fetch_records, normalize_row
API_ERROR_HELP = "Verify Airtable credentials (env or hardcoded) before redeploying."


def fetch_all_sources(sources: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []
    missing_sources = True
    for source in sources:
        table_id = source.get("table_id")
        base_id = source.get("base_id")
        if not table_id or not base_id:
            continue
        missing_sources = False
        label = source.get("label") or "Unknown"
        view_name = source.get("view")
        try:
            records = fetch_records(base_id=base_id, table_id=table_id, view_name=view_name)
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to pull data for {label}: {exc}") from exc
        rows.extend(normalize_row(record, talent=label) for record in records)

    if missing_sources:
        raise RuntimeError("No Airtable sources configured.")

    return rows


@st.cache_data(ttl=1800)
def load_bans() -> pd.DataFrame:
    if not SOURCES:
        raise RuntimeError("No Airtable sources configured.")

    df = pd.DataFrame(fetch_all_sources(SOURCES))
    if df.empty:
        return df
    for col in ("banned_at_local", "banned_at_utc"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    if df["banned_at_local"].dt.tz is not None:
        df["banned_at_local"] = df["banned_at_local"].dt.tz_convert(None)
    df["talent"] = df["talent"].fillna("Unknown")
    df["banned_date"] = df["banned_at_local"].dt.date
    if "previous_country" in df.columns:
        valid_countries = {"USA", "CAN", "AUS", "EU"}
        df["previous_country"] = df["previous_country"].apply(
            lambda val: str(val).strip().upper() if pd.notna(val) else None
        )
        df.loc[~df["previous_country"].isin(valid_countries), "previous_country"] = None
    if "previous_location" in df.columns:
        df["previous_location"] = df["previous_location"].apply(
            lambda val: val.strip() if isinstance(val, str) and val.strip() else None
        )
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
    default_range_index = range_options.index("Last 24 hours")
    range_choice = st.sidebar.selectbox(
        "Time range",
        range_options,
        index=default_range_index,
    )
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


def summary_metrics(df: pd.DataFrame):
    total_bans = len(df)
    now = pd.Timestamp.utcnow().tz_localize(None)
    df_times = df.dropna(subset=["banned_at_local"]).copy()
    if not df_times.empty and df_times["banned_at_local"].max() > now:
        now = df_times["banned_at_local"].max()

    today = now.date()
    start_of_today = pd.Timestamp(today)
    hours_today = max((now - start_of_today).total_seconds() / 3600, 1)
    window_24h = now - pd.Timedelta(hours=24)
    window_12h = now - pd.Timedelta(hours=12)
    window_4h = now - pd.Timedelta(hours=4)
    window_1h = now - pd.Timedelta(hours=1)

    todays_bans = df_times[df_times["banned_at_local"].dt.date == today].shape[0]
    last_24h = df_times[df_times["banned_at_local"] >= window_24h].shape[0]
    last_12h = df_times[df_times["banned_at_local"] >= window_12h].shape[0]
    last_4h = df_times[df_times["banned_at_local"] >= window_4h].shape[0]
    last_1h = df_times[df_times["banned_at_local"] >= window_1h].shape[0]

    avg_today = todays_bans / hours_today if todays_bans else 0
    avg_24h = last_24h / 24 if last_24h else 0
    avg_12h = last_12h / 12 if last_12h else 0
    avg_4h = last_4h / 4 if last_4h else 0
    avg_1h = last_1h  # one-hour window

    cols = st.columns(6)
    cols[0].metric("Total bans", f"{total_bans:,}")
    cols[1].metric("Today", f"{todays_bans:,}")
    cols[2].metric("Last 24h", f"{last_24h:,}")
    cols[3].metric("Last 12h", f"{last_12h:,}")
    cols[4].metric("Last 4h", f"{last_4h:,}")
    cols[5].metric("Last hour", f"{last_1h:,}")

    cols[1].caption(f"Avg/hr: {int(round(avg_today))}")
    cols[2].caption(f"Avg/hr: {int(round(avg_24h))}")
    cols[3].caption(f"Avg/hr: {int(round(avg_12h))}")
    cols[4].caption(f"Avg/hr: {int(round(avg_4h))}")
    cols[5].caption(f"Avg/hr: {int(round(avg_1h))}")

    if df.empty:
        return

    breakdown = (
        df.groupby("talent")
        .agg(bans=("account_id", "count"))
        .reset_index()
        .sort_values("bans", ascending=False)
    )

    if df_times.empty:
        today_counts = pd.Series(dtype="int64")
        last_24h_counts = pd.Series(dtype="int64")
        last_12h_counts = pd.Series(dtype="int64")
        last_4h_counts = pd.Series(dtype="int64")
        last_hour_counts = pd.Series(dtype="int64")
    else:
        today_counts = (
            df_times[df_times["banned_at_local"].dt.date == today]
            .groupby("talent")
            .size()
        )
        last_24h_counts = (
            df_times[df_times["banned_at_local"] >= window_24h]
            .groupby("talent")
            .size()
        )
        last_12h_counts = (
            df_times[df_times["banned_at_local"] >= window_12h]
            .groupby("talent")
            .size()
        )
        last_4h_counts = (
            df_times[df_times["banned_at_local"] >= window_4h]
            .groupby("talent")
            .size()
        )
        last_hour_counts = (
            df_times[df_times["banned_at_local"] >= window_1h]
            .groupby("talent")
            .size()
        )

    breakdown["Today"] = breakdown["talent"].map(today_counts).fillna(0).astype(int)
    breakdown["Last 24h"] = breakdown["talent"].map(last_24h_counts).fillna(0).astype(int)
    breakdown["Last 12h"] = breakdown["talent"].map(last_12h_counts).fillna(0).astype(int)
    breakdown["Last 4h"] = breakdown["talent"].map(last_4h_counts).fillna(0).astype(int)
    breakdown["Last hour"] = breakdown["talent"].map(last_hour_counts).fillna(0).astype(int)
    breakdown["bans"] = breakdown["bans"].astype(int)

    breakdown["Today Avg/hr"] = (breakdown["Today"] / hours_today).round().astype(int)
    breakdown["Last 24h Avg/hr"] = (breakdown["Last 24h"] / 24).round().astype(int)
    breakdown["Last 12h Avg/hr"] = (breakdown["Last 12h"] / 12).round().astype(int)
    breakdown["Last 4h Avg/hr"] = (breakdown["Last 4h"] / 4).round().astype(int)
    breakdown["Last hour Avg/hr"] = breakdown["Last hour"].round().astype(int)

    breakdown.rename(
        columns={
            "talent": "Model",
            "bans": "Bans",
        },
        inplace=True,
    )

    breakdown = breakdown[
        [
            "Model",
            "Bans",
            "Last 24h",
            "Last 24h Avg/hr",
            "Today",
            "Today Avg/hr",
            "Last 12h",
            "Last 12h Avg/hr",
            "Last 4h",
            "Last 4h Avg/hr",
            "Last hour",
            "Last hour Avg/hr",
        ]
    ]
    st.caption("Metrics reflect all bans (filters do not apply here).")
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
    period_title = "Date & time" if granularity == "Hourly" else "Date"
    time_format = "%b %d, %Y %H:%M" if granularity == "Hourly" else "%b %d, %Y"
    summary_counts = (
        daily.groupby("period", as_index=False)["count"]
        .sum()
        .rename(columns={"count": "total_bans"})
    )
    breakdown_strings = (
        daily.sort_values(["period", "count"], ascending=[True, False])
        .groupby("period")
        .apply(
            lambda grp: "\n".join(
                f"{talent or 'Unknown'}: {int(value)}"
                for talent, value in zip(grp["talent"], grp["count"])
            )
        )
        .reset_index(name="model_breakdown")
    )
    summary = summary_counts.merge(breakdown_strings, on="period", how="left")
    summary["model_breakdown"] = summary["model_breakdown"].fillna("No data")
    mask = summary["model_breakdown"] != "No data"
    summary.loc[mask, "model_breakdown"] = summary.loc[mask, "model_breakdown"].apply(
        lambda text: "\n".join(f"• {line}" for line in text.split("\n"))
    )
    delta = pd.Timedelta(days=1) if freq == "D" else pd.Timedelta(hours=1)
    summary["period_end"] = summary["period"] + delta
    daily = daily.merge(summary[["period", "total_bans"]], on="period", how="left")
    hover = alt.selection_point(
        fields=["period"], nearest=True, on="pointermove", empty="none"
    )

    color_scale = alt.Scale(
        range=[
            "#F97068",
            "#577590",
            "#43AA8B",
            "#F9C74F",
            "#9B5DE5",
            "#F9844A",
            "#90BE6D",
            "#277DA1",
            "#F94144",
            "#EF476F",
        ]
    )
    base = alt.Chart(daily)
    hover_region = (
        alt.Chart(summary)
        .mark_rect(opacity=0)
        .encode(
            x=alt.X("period:T", axis=None),
            x2=alt.X2("period_end:T"),
            tooltip=[
                alt.Tooltip("period:T", title=period_title, format=time_format),
                alt.Tooltip("total_bans:Q", title="Total bans"),
                alt.Tooltip("model_breakdown:N", title="By model"),
            ],
        )
        .add_params(hover)
    )
    lines = base.mark_line(
        point=alt.OverlayMarkDef(filled=True, size=50), strokeWidth=2
    ).encode(
        x=alt.X(
            "period:T",
            title=period_title,
            axis=alt.Axis(format=time_format if granularity == "Hourly" else "%b %d"),
        ),
        y=alt.Y("count:Q", title=f"Bans per {granularity.lower()}"),
        color=alt.Color(
            "talent:N",
            title="Model",
            scale=color_scale,
        ),
        tooltip=[
            alt.Tooltip("period:T", title=period_title, format=time_format),
            alt.Tooltip("talent:N", title="Model"),
            alt.Tooltip("count:Q", title=f"Bans ({granularity.lower()})"),
            alt.Tooltip("total_bans:Q", title="Total bans"),
        ],
    )
    lines = lines.add_params(hover)

    highlighted_points = (
        base.mark_circle(size=65)
        .encode(
            x=alt.X("period:T", title=period_title),
            y=alt.Y("count:Q", title=f"Bans per {granularity.lower()}"),
            color=alt.Color("talent:N", title="Model", scale=color_scale),
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        )
        .transform_filter(hover)
    )

    totals_rule = (
        alt.Chart(summary)
        .mark_rule(color="#A8A8A8")
        .encode(
            x=alt.X("period:T", title=period_title),
            tooltip=[
                alt.Tooltip("period:T", title=period_title, format=time_format),
                alt.Tooltip("total_bans:Q", title="Total bans"),
                alt.Tooltip("model_breakdown:N", title="By model"),
            ],
        )
        .transform_filter(hover)
    )

    chart = alt.layer(hover_region, lines, highlighted_points, totals_rule)
    st.altair_chart(chart.interactive(), use_container_width=True)


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
    model_palette = [
        "#F97068",
        "#577590",
        "#43AA8B",
        "#F9C74F",
        "#9B5DE5",
        "#F9844A",
        "#90BE6D",
        "#277DA1",
        "#F94144",
        "#EF476F",
    ]

    for idx, (column, label) in enumerate(chart_specs):
        if column not in df.columns:
            chart_columns[idx].info(f"No data for {label.lower()}.")
            continue
        data = (
            df.assign(category=df[column].fillna("Unknown"))
            .groupby("category")
            .size()
            .reset_index(name="Bans")
            .sort_values("Bans", ascending=False)
        )
        if data.empty:
            chart_columns[idx].info(f"No data for {label.lower()}.")
            continue
        data["Bans"] = pd.to_numeric(data["Bans"], errors="coerce").fillna(0)
        total_bans = data["Bans"].sum()
        if total_bans == 0:
            chart_columns[idx].info(f"No data for {label.lower()}.")
            continue
        data["Share"] = data["Bans"] / total_bans
        color_kwargs = {"title": label}
        if label == "Model":
            color_kwargs["scale"] = alt.Scale(range=model_palette)
        chart = (
            alt.Chart(data)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta("Bans:Q", stack=True),
                color=alt.Color("category:N", **color_kwargs),
                tooltip=[
                    alt.Tooltip("category:N", title=label),
                    alt.Tooltip("Bans:Q", title="Bans"),
                    alt.Tooltip("Share:Q", title="Share", format=".1%"),
                ],
            )
            .properties(title=label)
        )
        chart_columns[idx].altair_chart(chart, use_container_width=True)


def country_ban_rates(df: pd.DataFrame):
    st.subheader("Ban Rate by Country")
    if df.empty or "previous_country" not in df.columns or "previous_location" not in df.columns:
        st.info("Ban rate by country will appear once location data is available.")
        return

    total_locations_by_country = {"USA": 250, "CAN": 60, "AUS": 40, "EU": 150}
    valid_countries = {"USA", "CAN", "AUS", "EU"}
    data = df[df["previous_country"].isin(valid_countries)].copy()
    if data.empty:
        st.info("No country data available for the current filters.")
        return

    def count_locations(series: pd.Series) -> int:
        return series.dropna().nunique()

    grouped = (
        data.groupby(["talent", "previous_country"])
        .agg(bans=("account_id", "count"), seen_locations=("previous_location", count_locations))
        .reset_index()
    )
    overall = (
        data.groupby("previous_country")
        .agg(bans=("account_id", "count"), seen_locations=("previous_location", count_locations))
        .reset_index()
        .assign(talent="All models")
    )
    rates = pd.concat([overall, grouped], ignore_index=True, sort=False)
    if rates.empty:
        st.info("No country data available for the current filters.")
        return

    rates["locations"] = rates["previous_country"].map(total_locations_by_country).fillna(0).astype(int)
    rates["seen_locations"] = rates["seen_locations"].fillna(0).astype(int)
    rates["bans"] = rates["bans"].fillna(0).astype(int)
    rates["ban_rate"] = rates.apply(
        lambda row: row["bans"] / row["locations"] if row["locations"] else None, axis=1
    )
    rates["Model"] = rates["talent"].fillna("Unknown")
    rates["Country"] = rates["previous_country"]
    rates["Locations (cities)"] = rates["locations"]
    rates["Bans"] = rates["bans"]
    rates["Ban Rate"] = rates["ban_rate"].apply(lambda val: f"{val:.2f}" if val is not None else "N/A")
    rates["model_sort"] = rates["Model"].apply(lambda val: "" if val == "All models" else val)
    rates = rates.sort_values(["model_sort", "ban_rate"], ascending=[True, False])
    display_cols = ["Model", "Country", "Bans", "Ban Rate"]
    st.caption("Calculated as total bans in a country divided by fixed city counts (USA 250, CAN 60, AUS 40, EU 150).")

    overall = rates[rates["Model"] == "All models"]
    per_model = rates[rates["Model"] != "All models"]
    if not overall.empty:
        st.markdown("**All models**")
        st.dataframe(overall[display_cols], use_container_width=True, hide_index=True)
    if not per_model.empty:
        st.markdown("**Per model**")
        st.dataframe(per_model[display_cols], use_container_width=True, hide_index=True)


def top_locations_table(df: pd.DataFrame, limit: int | None = None):
    if df.empty or "previous_location" not in df.columns:
        return
    data = df.copy()
    data["previous_location"] = data["previous_location"].fillna("Unknown")
    grouped = (
        data.groupby(["previous_location", "previous_country"], dropna=False)
        .agg(bans=("account_id", "count"))
        .reset_index()
        .sort_values("bans", ascending=False)
    )
    if grouped.empty:
        return
    grouped = grouped.rename(
        columns={
            "previous_location": "Location",
            "previous_country": "Country",
            "bans": "Bans",
        }
    )
    title = "Locations by Bans" if limit is None else f"Top {limit} Locations by Bans"
    st.subheader(title)
    st.dataframe(grouped, use_container_width=True, hide_index=True)


def bans_table(df: pd.DataFrame):
    required_cols = {
        "banned_at_local": None,
        "talent": "Unknown",
        "account_id": None,
        "username": None,
        "email": None,
        "previous_location": None,
        "previous_country": None,
        "model": "Unknown",
        "platform": "Unknown",
        "reason": None,
        "about_me": None,
    }
    display_df = df.copy()
    for col, default in required_cols.items():
        if col not in display_df.columns:
            display_df[col] = default

    display_df = display_df[list(required_cols)].rename(
        columns={
            "banned_at_local": "Banned At",
            "talent": "Model",
            "account_id": "Account ID",
            "username": "Display Name",
            "previous_location": "Previous Location",
            "previous_country": "Previous Country",
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
    if st.button("Refresh Airtable data", type="primary"):
        load_bans.clear()
        st.rerun()

    st.caption("Explore bans pulled directly from Airtable bases.")

    try:
        df = load_bans()
    except RuntimeError as exc:
        st.error(f"{exc}\n{API_ERROR_HELP}")
        st.stop()
    if df.empty:
        st.info("No bans available yet. Confirm your NocoDB tables contain data.")
        return

    filtered_df, _ = sidebar_filters(df)

    summary_metrics(df)
    st.subheader("Trends")
    granularity_options = ["Daily", "Hourly"]
    granularity = st.radio(
        "Time granularity",
        granularity_options,
        horizontal=True,
        index=granularity_options.index("Hourly"),
    )
    time_series_chart(filtered_df, granularity)

    distribution_charts(filtered_df)
    country_ban_rates(filtered_df)
    top_locations_table(filtered_df, limit=20)
    st.subheader("Ban Details")
    bans_table(filtered_df)


if __name__ == "__main__":
    main()
