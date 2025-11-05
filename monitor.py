import os, sys, json, sqlite3, requests, datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

BASE_URL = os.getenv("NOCODB_BASE_URL", "https://nocodb.fanworkz.com/")
API_TOKEN = os.getenv("NOCODB_API_TOKEN", "_U5A02swoTddDJDEAMN5g2gaUg8rFe4IKAtWNh4e")
DEFAULT_EMILY_TABLE = os.getenv("NOCODB_TABLE_ID_EMILY", "m5uejvzfy8wa2ku")
DEFAULT_EMILY_VIEW = os.getenv("NOCODB_VIEW_ID_EMILY", "vwxzmxmeludlbp1r")
DEFAULT_ERIC_TABLE = os.getenv("NOCODB_TABLE_ID_ERIC", "mwr6v21uxqu16yh")
DEFAULT_ERIC_VIEW = os.getenv("NOCODB_VIEW_ID_ERIC", "vwgqx8nx1e149t0k")
APP_TZ    = os.getenv("APP_TZ", "UTC")

DB_PATH = Path("data/bans.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

HEADERS = {"xc-token": API_TOKEN, "accept": "application/json"}

SOURCES = [
    {"label": "Emily", "table_id": DEFAULT_EMILY_TABLE, "view_id": DEFAULT_EMILY_VIEW},
    {"label": "Eric", "table_id": DEFAULT_ERIC_TABLE, "view_id": DEFAULT_ERIC_VIEW},
]
LEGACY_TABLE_ID = os.getenv("NOCODB_TABLE_ID")
LEGACY_VIEW_ID = os.getenv("NOCODB_VIEW_ID")
if LEGACY_TABLE_ID:
    SOURCES.insert(0, {"label": "Default", "table_id": LEGACY_TABLE_ID, "view_id": LEGACY_VIEW_ID})
SOURCES = [s for s in SOURCES if s.get("table_id")]

# ---- Adjust this map to your real NocoDB column names ----
FIELD_MAP = {
    "account_id": ["account_id", "id", "acc_id"],
    "username":   ["username", "user", "handle", "Display Name", "Dislpay Name"],
    "email":      ["email", "Email"],
    "about_me":   ["about_me", "bio", "About Me"],
    "model":      ["model", "Associated Model", "OnlyFans model", "device_model", "Device Model"],
    "platform":   ["platform", "source_app", "app", "device_manufacturer", "Device Manufacturer"],
    "reason":     ["reason", "ban_reason", "Ban Reason"],
    "status":     ["status"],
    "banned_at":  ["banned_at", "created_at", "updated_at", "Ban Time"],
}
DEFAULTS = {
    "model": "Unknown",
    "platform": "Unknown",
    "reason": "Unknown",
    "status": "banned",
}

def pick(d, keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return DEFAULTS.get(keys[0], default)

def fetch_records(table_id, limit=1000, view_id=None):
    items, offset = [], 0
    while True:
        params = {"limit": limit, "offset": offset}
        if view_id:
            params["viewId"] = view_id
        url = f"{BASE_URL}/api/v2/tables/{table_id}/records"
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        rows = data.get("list", data if isinstance(data, list) else [])
        if not rows: break
        items.extend(rows)
        if len(rows) < limit: break
        offset += limit
    return items

def parse_dt_iso(s):
    if not s: return None
    s = s.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None

def to_local(ts_utc):
    if ts_utc is None: return None
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=dt.timezone.utc)
    return ts_utc.astimezone(ZoneInfo(APP_TZ))

def normalize_row(r, talent=None):
    row = {
        "account_id": pick(r, FIELD_MAP["account_id"]),
        "username":   pick(r, FIELD_MAP["username"]),
        "email":      pick(r, FIELD_MAP["email"]),
        "about_me":   pick(r, FIELD_MAP["about_me"]),
        "model":      pick(r, FIELD_MAP["model"], "Unknown"),
        "platform":   pick(r, FIELD_MAP["platform"], "Unknown"),
        "reason":     pick(r, FIELD_MAP["reason"], "Unknown"),
        "status":     pick(r, FIELD_MAP["status"], "banned"),
        "banned_at":  pick(r, FIELD_MAP["banned_at"]),
        "talent":     talent or "Unknown",
    }
    if not row["account_id"]:
        row["account_id"] = row["email"] or row["username"]

    ts = parse_dt_iso(row["banned_at"])
    row["banned_at_utc"] = ts.isoformat() if ts else None
    row["banned_at_local"] = to_local(ts).isoformat() if ts else None
    return row

def ensure_columns(cur, table, columns):
    existing = {info[1] for info in cur.execute(f"PRAGMA table_info({table});")}
    for name, ddl in columns.items():
        if name not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl};")
            existing.add(name)
    return existing

def init_db(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS bans (
      account_id TEXT,
      username   TEXT,
      email      TEXT,
      about_me   TEXT,
      model      TEXT,
      platform   TEXT,
      reason     TEXT,
      status     TEXT,
      talent     TEXT,
      banned_at_utc   TEXT,
      banned_at_local TEXT,
      PRIMARY KEY (account_id, banned_at_utc)
    );
    """)
    columns = ensure_columns(cur, "bans", {
        "email": "TEXT",
        "about_me": "TEXT",
        "talent": "TEXT",
    })
    if "talent" in columns:
        cur.execute("UPDATE bans SET talent = COALESCE(talent, 'Unknown') WHERE talent IS NULL;")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bans_banned_at_local ON bans(banned_at_local);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bans_model ON bans(model);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bans_reason ON bans(reason);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bans_platform ON bans(platform);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bans_talent ON bans(talent);")

    # hourly/day aggregates
    cur.execute("DROP TABLE IF EXISTS agg_hourly;")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS agg_hourly (
      hour_local TEXT,    -- e.g. 2025-11-05T14:00:00+01:00
      talent     TEXT,
      model      TEXT,
      reason     TEXT,
      platform   TEXT,
      count      INTEGER,
      PRIMARY KEY (hour_local, talent, model, reason, platform)
    );
    """)
    cur.execute("DROP TABLE IF EXISTS agg_daily;")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS agg_daily (
      day_local  TEXT,    -- e.g. 2025-11-05
      talent     TEXT,
      model      TEXT,
      reason     TEXT,
      platform   TEXT,
      count      INTEGER,
      PRIMARY KEY (day_local, talent, model, reason, platform)
    );
    """)
    conn.commit()

def upsert_bans(conn, rows):
    cur = conn.cursor()
    if not rows:
        return 0
    q = """INSERT OR IGNORE INTO bans
           (account_id, username, email, about_me, model, platform, reason, status, talent, banned_at_utc, banned_at_local)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""
    data = []
    for r in rows:
        data.append((
            r["account_id"], r["username"], r["email"], r["about_me"],
            r["model"], r["platform"], r["reason"], r["status"], r["talent"],
            r["banned_at_utc"], r["banned_at_local"]
        ))
    cur.executemany(q, data)
    inserted = cur.rowcount
    cur.executemany(
        """
        UPDATE bans
        SET talent = ?
        WHERE (account_id = ?)
          AND (banned_at_utc = ? OR banned_at_utc IS NULL)
        """,
        [
            (r["talent"], r["account_id"], r["banned_at_utc"])
            for r in rows
            if r["account_id"]
        ],
    )
    conn.commit()
    return inserted  # newly inserted rows

def rebuild_aggregates(conn):
    cur = conn.cursor()
    # recompute from scratch; small table so OK. For huge data, window by recent time.
    cur.execute("DELETE FROM agg_hourly;")
    cur.execute("DELETE FROM agg_daily;")

    # Hourly
    cur.execute("""
    INSERT INTO agg_hourly (hour_local, talent, model, reason, platform, count)
    SELECT
      strftime('%Y-%m-%dT%H:00:00', banned_at_local) || substr(banned_at_local, 20) AS hour_local,
      COALESCE(talent,'Unknown'),
      COALESCE(model,'Unknown'), COALESCE(reason,'Unknown'), COALESCE(platform,'Unknown'),
      COUNT(*)
    FROM bans
    WHERE banned_at_local IS NOT NULL
    GROUP BY 1,2,3,4,5;
    """)

    # Daily
    cur.execute("""
    INSERT INTO agg_daily (day_local, talent, model, reason, platform, count)
    SELECT
      substr(banned_at_local,1,10) as day_local,
      COALESCE(talent,'Unknown'),
      COALESCE(model,'Unknown'), COALESCE(reason,'Unknown'), COALESCE(platform,'Unknown'),
      COUNT(*)
    FROM bans
    WHERE banned_at_local IS NOT NULL
    GROUP BY 1,2,3,4,5;
    """)
    conn.commit()

def main():
    if not (BASE_URL and API_TOKEN):
        print("Missing env vars NOCODB_BASE_URL or NOCODB_API_TOKEN", file=sys.stderr)
        sys.exit(1)
    if not SOURCES:
        print("No NocoDB sources configured.", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    for source in SOURCES:
        table_id = source.get("table_id")
        if not table_id:
            continue
        label = source.get("label") or "Unknown"
        view_id = source.get("view_id")
        records = fetch_records(table_id=table_id, view_id=view_id)
        rows = [normalize_row(r, talent=label) for r in records]
        all_rows.extend(rows)
        print(f"[INFO] Loaded {len(rows)} rows for {label}")

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    inserted = upsert_bans(conn, all_rows)
    rebuild_aggregates(conn)
    conn.close()
    print(f"[OK] Upserted new rows: {inserted} | Aggregates rebuilt at {dt.datetime.now().isoformat()}")

if __name__ == "__main__":
    main()
