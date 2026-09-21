"""
OpsPulse - Camada 3 (Modelagem) - Prophet
Previsão de volume diário de incidentes P2 e P3, D+1 e D+7,
com sazonalidade semanal e mensal (identificada na EDA da Sprint 2).
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from data_prep import load_raw, clean_and_engineer, daily_counts_by_priority

FORECAST_HORIZON_DAYS = 7


def train_and_forecast_priority(daily: pd.DataFrame, priority: str) -> tuple[pd.DataFrame, dict]:
    """Treina um Prophet para uma prioridade e retorna o forecast + métricas de validação."""
    sub = daily[daily["prioridade_label"] == priority][["data_abertura", "incidentes"]]
    sub = sub.rename(columns={"data_abertura": "ds", "incidentes": "y"}).sort_values("ds")

    # Holdout dos últimos 14 dias para validar MAPE (métrica citada na proposta de solução)
    cutoff = sub["ds"].max() - pd.Timedelta(days=14)
    train, test = sub[sub["ds"] <= cutoff], sub[sub["ds"] > cutoff]

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        interval_width=0.8,
    )
    model.add_country_holidays(country_name="BR")
    model.fit(train)

    future = model.make_future_dataframe(periods=14 + FORECAST_HORIZON_DAYS)
    forecast = model.predict(future)

    # MAPE no holdout
    merged = test.merge(forecast[["ds", "yhat"]], on="ds", how="left")
    mape = float(np.mean(np.abs((merged["y"] - merged["yhat"]) / merged["y"].replace(0, np.nan))) * 100)
    mae = float(np.mean(np.abs(merged["y"] - merged["yhat"])))

    # Reajusta com a base completa para gerar o forecast final D+1..D+7
    model_full = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        interval_width=0.8,
    )
    model_full.add_country_holidays(country_name="BR")
    model_full.fit(sub)
    future_full = model_full.make_future_dataframe(periods=FORECAST_HORIZON_DAYS)
    forecast_full = model_full.predict(future_full)
    forecast_full["prioridade"] = priority

    metrics = {"priority": priority, "mape": round(mape, 2), "mae": round(mae, 1), "n_days": len(sub)}
    return forecast_full[["ds", "yhat", "yhat_lower", "yhat_upper", "prioridade"]], metrics


def main():
    raw = load_raw()
    clean = clean_and_engineer(raw)
    daily = daily_counts_by_priority(clean, priorities=("P2", "P3"))

    all_forecasts = []
    all_metrics = []
    for priority in ("P2", "P3"):
        fc, metrics = train_and_forecast_priority(daily, priority)
        all_forecasts.append(fc)
        all_metrics.append(metrics)
        print(f"{priority}: MAPE={metrics['mape']}% | MAE={metrics['mae']} inc/dia ({metrics['n_days']} dias de histórico)")

    result = pd.concat(all_forecasts, ignore_index=True)
    result.to_csv("models/prophet_forecast.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("models/prophet_metrics.csv", index=False)

    daily.to_csv("models/daily_history.csv", index=False)
    print("\nForecast salvo em models/prophet_forecast.csv")


if __name__ == "__main__":
    main()
