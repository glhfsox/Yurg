BEGIN;

CREATE TABLE IF NOT EXISTS coin (
  coin_id SERIAL PRIMARY KEY,
  symbol TEXT UNIQUE NOT NULL
);

INSERT INTO coin(symbol)
VALUES ('BTCUSDT'), ('ETHUSDT')
ON CONFLICT (symbol) DO NOTHING;

CREATE TABLE IF NOT EXISTS ohlcv_15m (
  id BIGSERIAL PRIMARY KEY,
  coin_id INTEGER NOT NULL REFERENCES coin(coin_id),
  ts TIMESTAMPTZ NOT NULL,
  open NUMERIC(18,8) NOT NULL,
  high NUMERIC(18,8) NOT NULL,
  low NUMERIC(18,8) NOT NULL,
  close NUMERIC(18,8) NOT NULL,
  volume NUMERIC(30,8) NOT NULL,
  UNIQUE (coin_id, ts)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_15m_coin_ts ON ohlcv_15m (coin_id, ts);

CREATE TABLE IF NOT EXISTS ohlcv_1h (
  id BIGSERIAL PRIMARY KEY,
  coin_id INTEGER NOT NULL REFERENCES coin(coin_id),
  ts TIMESTAMPTZ NOT NULL,
  open NUMERIC(18,8) NOT NULL,
  high NUMERIC(18,8) NOT NULL,
  low NUMERIC(18,8) NOT NULL,
  close NUMERIC(18,8) NOT NULL,
  volume NUMERIC(30,8) NOT NULL,
  UNIQUE (coin_id, ts)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_1h_coin_ts ON ohlcv_1h (coin_id, ts);

CREATE TABLE IF NOT EXISTS features_15m (
  id BIGSERIAL PRIMARY KEY,
  coin_id INTEGER NOT NULL REFERENCES coin(coin_id),
  ts TIMESTAMPTZ NOT NULL,

  ret_1 DOUBLE PRECISION,
  ret_3 DOUBLE PRECISION,
  ret_6 DOUBLE PRECISION,
  ret_12 DOUBLE PRECISION,
  ret_24 DOUBLE PRECISION,
  vol_ret_6 DOUBLE PRECISION,
  vol_ret_24 DOUBLE PRECISION,
  high_low_range_6 DOUBLE PRECISION,
  close_to_high DOUBLE PRECISION,
  close_to_low DOUBLE PRECISION,

  ma_8 DOUBLE PRECISION,
  ma_32 DOUBLE PRECISION,
  ma_96 DOUBLE PRECISION,
  close_ma8_ratio DOUBLE PRECISION,
  close_ma32_ratio DOUBLE PRECISION,
  ma8_ma32_ratio DOUBLE PRECISION,
  ma32_ma96_ratio DOUBLE PRECISION,
  ma8_diff_4 DOUBLE PRECISION,
  ma32_diff_8 DOUBLE PRECISION,

  vol_1 DOUBLE PRECISION,
  vol_8_mean DOUBLE PRECISION,
  vol_32_mean DOUBLE PRECISION,
  vol_8_std DOUBLE PRECISION,
  vol_zscore_8 DOUBLE PRECISION,

  hour_ret_1 DOUBLE PRECISION,
  hour_ret_4 DOUBLE PRECISION,
  hour_ret_24 DOUBLE PRECISION,
  hour_vol_ret_24 DOUBLE PRECISION,
  hour_vol_zscore_24 DOUBLE PRECISION,

  hour_of_day SMALLINT,
  day_of_week SMALLINT,
  is_weekend SMALLINT,
  sin_hour DOUBLE PRECISION,
  cos_hour DOUBLE PRECISION,

  other_ret_1 DOUBLE PRECISION,
  other_ret_6 DOUBLE PRECISION,
  other_close_ma32_ratio DOUBLE PRECISION,
  diff_ret_1 DOUBLE PRECISION,
  diff_close_ma32_ratio DOUBLE PRECISION,

  target_class SMALLINT CHECK (target_class IN (-1, 0, 1)),
  UNIQUE (coin_id, ts)
);

CREATE INDEX IF NOT EXISTS idx_features_15m_coin_ts ON features_15m (coin_id, ts);

CREATE TABLE IF NOT EXISTS predictions_15m (
  id BIGSERIAL PRIMARY KEY,
  coin_id INTEGER NOT NULL REFERENCES coin(coin_id),
  ts TIMESTAMPTZ NOT NULL,
  model_version TEXT NOT NULL,
  predicted_class SMALLINT NOT NULL CHECK (predicted_class IN (-1, 0, 1)),
  prob_down DOUBLE PRECISION NOT NULL,
  prob_flat DOUBLE PRECISION NOT NULL,
  prob_up DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (coin_id, ts, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_15m_coin_ts ON predictions_15m (coin_id, ts);
CREATE INDEX IF NOT EXISTS idx_predictions_15m_model_version ON predictions_15m (model_version);

COMMIT;
