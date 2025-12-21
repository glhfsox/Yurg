CREATE OR REPLACE VIEW v_daily_summary AS
WITH daily AS (
  SELECT
    coin_id,
    date_trunc('day', ts) AS date,
    (array_agg(open ORDER BY ts ASC))[1] AS daily_open,
    MAX(high) AS daily_high,
    MIN(low) AS daily_low,
    (array_agg(close ORDER BY ts DESC))[1] AS daily_close,
    SUM(volume) AS daily_volume
  FROM ohlcv_15m
  GROUP BY coin_id, date_trunc('day', ts)
)
SELECT
  coin_id,
  date,
  daily_open,
  daily_high,
  daily_low,
  daily_close,
  daily_volume,
  ln(daily_close) - ln(lag(daily_close) OVER (PARTITION BY coin_id ORDER BY date)) AS daily_return
FROM daily;

CREATE OR REPLACE VIEW v_volatility_rolling AS
SELECT
  coin_id,
  date,
  daily_return,
  STDDEV(daily_return) OVER (
    PARTITION BY coin_id
    ORDER BY date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS vol_7d,
  STDDEV(daily_return) OVER (
    PARTITION BY coin_id
    ORDER BY date
    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
  ) AS vol_30d
FROM v_daily_summary
WHERE daily_return IS NOT NULL;

CREATE OR REPLACE VIEW v_signal_distribution AS
SELECT
  p.coin_id,
  date_trunc('week', p.ts) AS week,
  COUNT(*) FILTER (WHERE f.target_class = -1) AS actual_down,
  COUNT(*) FILTER (WHERE f.target_class = 0) AS actual_flat,
  COUNT(*) FILTER (WHERE f.target_class = 1) AS actual_up,
  COUNT(*) FILTER (WHERE p.predicted_class = -1) AS predicted_down,
  COUNT(*) FILTER (WHERE p.predicted_class = 0) AS predicted_flat,
  COUNT(*) FILTER (WHERE p.predicted_class = 1) AS predicted_up,
  AVG((p.predicted_class = f.target_class)::INT)::DOUBLE PRECISION AS accuracy
FROM predictions_15m p
JOIN features_15m f
  ON f.coin_id = p.coin_id
  AND f.ts = p.ts
WHERE f.target_class IS NOT NULL
GROUP BY p.coin_id, date_trunc('week', p.ts);
