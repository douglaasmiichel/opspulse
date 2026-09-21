"""
OpsPulse - Camada 2 (Processamento)
Limpeza, tratamento de nulos e criação de features temporais
a partir do LW-DATASET.xlsx (Camada 1 - Ingestão).
"""
import pandas as pd
import numpy as np

DATASET_PATH = "data/LW-DATASET.xlsx"

DIAS_SEMANA_PT = {
    0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta",
    4: "Sexta", 5: "Sábado", 6: "Domingo",
}


def load_raw(path: str = DATASET_PATH) -> pd.DataFrame:
    return pd.read_excel(path)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza + criação de features temporais, conforme Camada 2 da arquitetura."""
    df = df.copy()

    # Prioridade -> código numérico e rótulo curto (P1..P5)
    df["prioridade_num"] = df["Prioridade"].str.extract(r"^(\d)").astype(int)
    df["prioridade_label"] = "P" + df["prioridade_num"].astype(str)

    # Features temporais a partir da abertura do incidente
    df["data_abertura"] = df["Aberto"].dt.date
    df["dia_semana_num"] = df["Aberto"].dt.dayofweek
    df["dia_semana"] = df["dia_semana_num"].map(DIAS_SEMANA_PT)
    df["hora_abertura"] = df["Aberto"].dt.hour
    df["mes_abertura"] = df["Aberto"].dt.to_period("M").astype(str)
    df["dia_util"] = df["dia_semana_num"] < 5

    # Tratamento de nulos em categóricas (produto/categoria têm muitos nulos:
    # incidentes de monitoramento de infraestrutura sem produto associado)
    for col in ["Produto", "Categoria", "Subcategoria"]:
        df[col] = df[col].fillna("Não especificado")

    # Alvo de risco de OLA: só faz sentido para incidentes que entraram
    # na apuração de KPI (~21% da base). Fora desse universo não há rótulo.
    df["entrou_kpi"] = (df["Entrou para KPI?"] == "SIM")
    df["kpi_violado"] = (df["KPI Violado?"] == "SIM").astype(int)

    return df


def daily_counts_by_priority(df: pd.DataFrame, priorities=("P2", "P3")) -> pd.DataFrame:
    """Série diária de volume de incidentes por prioridade, para o Prophet."""
    sub = df[df["prioridade_label"].isin(priorities)].copy()
    daily = (
        sub.groupby(["data_abertura", "prioridade_label"])
        .size()
        .reset_index(name="incidentes")
    )
    daily["data_abertura"] = pd.to_datetime(daily["data_abertura"])
    return daily


def kpi_risk_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Subconjunto rotulado (entrou_kpi=True) com as features para o XGBoost."""
    return df[df["entrou_kpi"]].copy()


if __name__ == "__main__":
    raw = load_raw()
    clean = clean_and_engineer(raw)
    print(f"Registros: {len(clean):,}")
    print(clean["prioridade_label"].value_counts())
    print(f"\nEntraram para KPI: {clean['entrou_kpi'].sum():,}")
    print(f"Violações de KPI: {clean['kpi_violado'].sum():,}")
