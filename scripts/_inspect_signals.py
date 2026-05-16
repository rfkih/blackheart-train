"""One-shot inspection of signal_history + model_registry current state.
Not part of the test suite — ad-hoc operator query.
"""
import psycopg

CONN = "host=127.0.0.1 port=5432 dbname=trading_db user=blackheart_research password=research_dev_pass"


def main() -> None:
    with psycopg.connect(CONN) as conn:
        cur = conn.cursor()

        print("=== signal_history columns ===")
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name='signal_history' ORDER BY ordinal_position"
        )
        cols = cur.fetchall()
        for c in cols:
            print(f"  {c[0]:30s} {c[1]}")

        # Try the common shape first; fall back to discovery if it errors.
        col_names = [c[0] for c in cols]
        signal_id_col = next(
            (c for c in col_names if c in ("signal_id", "model_name", "name", "spec_name")),
            None,
        )
        ts_col = next((c for c in col_names if c in ("ts", "as_of", "timestamp")), None)
        value_col = next(
            (c for c in col_names if c in ("value", "prediction", "score", "proba")),
            None,
        )
        symbol_col = "symbol" if "symbol" in col_names else None

        print(f"\nusing columns: id={signal_id_col} ts={ts_col} value={value_col} symbol={symbol_col}")

        print("\n=== rows per (id, symbol) ===")
        sql = f"SELECT {signal_id_col}"
        if symbol_col:
            sql += f", {symbol_col}"
        sql += f", COUNT(*), MIN({ts_col})::date, MAX({ts_col})::date FROM signal_history GROUP BY 1"
        if symbol_col:
            sql += ", 2"
        sql += " ORDER BY 1"
        cur.execute(sql)
        for r in cur.fetchall():
            print(f"  {r}")

        print("\n=== latest 5 predictions per (id, symbol) ===")
        sql = (
            f"SELECT {signal_id_col}"
            + (f", {symbol_col}" if symbol_col else "")
            + f", {ts_col}, {value_col} FROM signal_history "
            f"ORDER BY {ts_col} DESC LIMIT 10"
        )
        cur.execute(sql)
        for r in cur.fetchall():
            print(f"  {r}")

        print("\n=== prediction distribution (last 90 days) ===")
        sql = (
            f"SELECT {signal_id_col}, "
            f"COUNT(*) AS n, "
            f"AVG({value_col})::numeric(6,4) AS mean, "
            f"MIN({value_col})::numeric(6,4) AS min, "
            f"MAX({value_col})::numeric(6,4) AS max, "
            f"SUM(CASE WHEN {value_col} < 0.30 THEN 1 ELSE 0 END) AS n_low, "
            f"SUM(CASE WHEN {value_col} > 0.70 THEN 1 ELSE 0 END) AS n_high "
            f"FROM signal_history WHERE {ts_col} >= NOW() - INTERVAL '90 days' "
            f"GROUP BY {signal_id_col} ORDER BY {signal_id_col}"
        )
        cur.execute(sql)
        for r in cur.fetchall():
            print(f"  {r}")

        print("\n=== signal_history by source ===")
        cur.execute(
            "SELECT source, COUNT(*), MIN(ts)::date, MAX(ts)::date, "
            "AVG(value)::numeric(6,4), MIN(value)::numeric(6,4), MAX(value)::numeric(6,4) "
            "FROM signal_history GROUP BY source ORDER BY MAX(ts) DESC"
        )
        for r in cur.fetchall():
            print(f"  source={r[0]} n={r[1]} range={r[2]}..{r[3]} mean={r[4]} min={r[5]} max={r[6]}")

        print("\n=== latest 10 predictions (any source) with meta ===")
        cur.execute(
            "SELECT ts, value, confidence, source, meta FROM signal_history "
            "ORDER BY ts DESC LIMIT 10"
        )
        for r in cur.fetchall():
            ts, val, conf, src, meta = r
            print(f"  {ts.isoformat()} value={val:.4f} conf={conf} source={src} meta={meta}")

        print("\n=== signal interpretation ===")
        cur.execute(
            "SELECT ts, value FROM signal_history ORDER BY ts DESC LIMIT 1"
        )
        latest = cur.fetchone()
        if latest:
            v = latest[1]
            label = (
                "RISK-OFF (gate would BLOCK longs, ALLOW shorts)" if v < 0.30
                else "RISK-ON (gate would ALLOW longs, BLOCK shorts)" if v > 0.70
                else "NEUTRAL (gate would NOT veto)"
            )
            print(f"  latest @ {latest[0].isoformat()}: value={v:.4f} → {label}")

        print("\n=== model_registry columns ===")
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name='model_registry' ORDER BY ordinal_position"
        )
        for c in cur.fetchall():
            print(f"  {c[0]:35s} {c[1]}")

        print("\n=== model_registry rows ===")
        cur.execute("SELECT * FROM model_registry ORDER BY 1 LIMIT 10")
        col_names = [d.name for d in cur.description]
        for r in cur.fetchall():
            print("  ---")
            for k, v in zip(col_names, r):
                if isinstance(v, str) and len(v) > 80:
                    v = v[:60] + "..."
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
