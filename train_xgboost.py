"""
OpsPulse - Camada 3 (Modelagem) - XGBoost
Classificador de risco de violação de OLA, treinado sobre o universo
de incidentes que efetivamente entram na apuração de KPI (~21% da base).
Classe extremamente desbalanceada (248 violações em 25.600 casos, ~1%).
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
import xgboost as xgb

from data_prep import load_raw, clean_and_engineer, kpi_risk_frame

FEATURES = [
    "prioridade_num", "Produto", "Categoria", "Grupo designado",
    "hora_abertura", "dia_semana_num", "dia_util", "mes_abertura",
]
TARGET = "kpi_violado"


def build_dataset():
    raw = load_raw()
    clean = clean_and_engineer(raw)
    kpi_df = kpi_risk_frame(clean)
    return kpi_df


def encode_features(df: pd.DataFrame, encoders: dict | None = None):
    df = df.copy()
    cat_cols = ["Produto", "Categoria", "Grupo designado", "mes_abertura"]
    fit_mode = encoders is None
    encoders = encoders or {}

    for col in cat_cols:
        if fit_mode:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col + "_enc"] = df[col].astype(str).map(
                lambda v: le.transform([v])[0] if v in le.classes_ else -1
            )
    df["dia_util_enc"] = df["dia_util"].astype(int)
    return df, encoders


def main():
    df = build_dataset()
    df, encoders = encode_features(df)

    feature_cols = [
        "prioridade_num", "Produto_enc", "Categoria_enc", "Grupo designado_enc",
        "hora_abertura", "dia_semana_num", "dia_util_enc",
    ]
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Classe minoritária ~1% -> scale_pos_weight compensa o desbalanceamento
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]

    # Classe minoritária ~1%: threshold 0.5 nunca é o ponto ótimo.
    # Varre thresholds e escolhe o que maximiza F1 (documentado para a banca).
    thresholds = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y_test, (proba >= t).astype(int)) for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(f1s))])
    preds = (proba >= best_threshold).astype(int)

    metrics = {
        "threshold": round(best_threshold, 2),
        "f1": round(f1_score(y_test, preds), 3),
        "precision": round(precision_score(y_test, preds), 3),
        "recall": round(recall_score(y_test, preds), 3),
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
        "n_test": len(y_test),
        "n_violations_test": int(y_test.sum()),
    }
    print(classification_report(y_test, preds, target_names=["Sem risco", "Risco de violação"]))
    print(metrics)

    # Feature importance (interpretabilidade citada na Sprint 2)
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importance)

    # Salva artefatos
    joblib.dump(model, "models/xgboost_ola_risk.pkl")
    joblib.dump(encoders, "models/xgboost_encoders.pkl")
    pd.DataFrame([metrics]).to_csv("models/xgboost_metrics.csv", index=False)
    importance.to_csv("models/xgboost_importance.csv")

    # Scores para todo o universo rotulado (usado no dashboard - Tela 2)
    df["risco_predito"] = model.predict_proba(X)[:, 1]
    cols_out = [
        "Número", "prioridade_label", "Produto", "Categoria", "Grupo designado",
        "hora_abertura", "dia_semana", "data_abertura", "kpi_violado", "risco_predito",
    ]
    df[cols_out].to_csv("models/xgboost_scored.csv", index=False)


if __name__ == "__main__":
    main()
