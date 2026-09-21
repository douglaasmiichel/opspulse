"""
OpsPulse - Camada 4 (Visualização)
Dashboard Streamlit + Plotly com as 4 telas prototipadas na Sprint 2:
1. Previsão D+1/D+7 (Prophet)
2. Risco de OLA (XGBoost)
3. Análise Exploratória Interativa
4. Recomendações Operacionais
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_prep import load_raw, clean_and_engineer

st.set_page_config(page_title="OpsPulse | Locaweb Challenge 2026", layout="wide", page_icon="📡")


@st.cache_data
def get_clean_data():
    raw = load_raw()
    return clean_and_engineer(raw)


@st.cache_data
def get_forecast():
    return pd.read_csv("models/prophet_forecast.csv", parse_dates=["ds"])


@st.cache_data
def get_history():
    return pd.read_csv("models/daily_history.csv", parse_dates=["data_abertura"])


@st.cache_data
def get_prophet_metrics():
    return pd.read_csv("models/prophet_metrics.csv")


@st.cache_data
def get_scored():
    return pd.read_csv("models/xgboost_scored.csv", parse_dates=["data_abertura"])


@st.cache_data
def get_xgb_metrics():
    return pd.read_csv("models/xgboost_metrics.csv")


df = get_clean_data()
forecast = get_forecast()
history = get_history()
prophet_metrics = get_prophet_metrics()
scored = get_scored()
xgb_metrics = get_xgb_metrics()

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title("📡 OpsPulse")
st.sidebar.caption("AIOps para Previsão de Incidentes de TI — Locaweb Challenge 2026")
st.sidebar.markdown("**Grupo:** Os Mosqueteiros — Turma 2TSCOA")
modo = st.sidebar.radio("Modo de visualização", ["Executivo", "Técnico"], index=0)
tela = st.sidebar.radio(
    "Navegação",
    ["1. Previsão D+1/D+7", "2. Risco de OLA", "3. Análise Exploratória", "4. Recomendações"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Base: {len(df):,} incidentes | Jan/2023 – Dez/2025")

# ------------------------------------------------------------------
# TELA 1 — Previsão D+1 e D+7
# ------------------------------------------------------------------
if tela.startswith("1"):
    st.title("Previsão D+1 e D+7 — Volume de Incidentes")
    st.caption("Modelo Prophet | Prioridades P2 (Alta) e P3 (Média)")

    col_f1, col_f2 = st.columns(2)
    prioridade_sel = col_f1.selectbox("Prioridade", ["P2", "P3"])
    dias_hist = col_f2.slider("Dias de histórico exibidos", 30, 365, 90)

    hist_p = history[history["prioridade_label"] == prioridade_sel].sort_values("data_abertura")
    fc_p = forecast[forecast["prioridade"] == prioridade_sel].sort_values("ds")
    hoje = hist_p["data_abertura"].max()
    fc_futuro = fc_p[fc_p["ds"] > hoje]

    d1 = fc_futuro.iloc[0] if len(fc_futuro) > 0 else None
    d7 = fc_futuro.iloc[6] if len(fc_futuro) > 6 else None
    media_hist = hist_p["incidentes"].tail(30).mean()

    c1, c2, c3 = st.columns(3)
    with c1:
        if d1 is not None:
            delta = d1["yhat"] - media_hist
            cor = "🔴" if delta > media_hist * 0.3 else ("🟡" if delta > 0 else "🟢")
            st.metric(f"{cor} Previsão D+1", f"{d1['yhat']:.0f} incidentes", f"{delta:+.0f} vs média 30d")
    with c2:
        if d7 is not None:
            delta7 = d7["yhat"] - media_hist
            cor7 = "🔴" if delta7 > media_hist * 0.3 else ("🟡" if delta7 > 0 else "🟢")
            st.metric(f"{cor7} Previsão D+7", f"{d7['yhat']:.0f} incidentes", f"{delta7:+.0f} vs média 30d")
    with c3:
        mape_row = prophet_metrics[prophet_metrics["priority"] == prioridade_sel]
        mape_val = mape_row["mape"].iloc[0] if len(mape_row) else np.nan
        st.metric("MAPE (validação, 14d)", f"{mape_val:.1f}%")

    # Gráfico temporal: histórico + previsão com intervalo de confiança
    fig = go.Figure()
    hist_plot = hist_p.tail(dias_hist)
    fig.add_trace(go.Scatter(x=hist_plot["data_abertura"], y=hist_plot["incidentes"],
                              name="Histórico real", mode="lines", line=dict(color="#3B4CCA")))
    fig.add_trace(go.Scatter(x=fc_futuro["ds"], y=fc_futuro["yhat"],
                              name="Previsão Prophet", mode="lines+markers", line=dict(color="#E8590C", dash="dash")))
    fig.add_trace(go.Scatter(
        x=list(fc_futuro["ds"]) + list(fc_futuro["ds"][::-1]),
        y=list(fc_futuro["yhat_upper"]) + list(fc_futuro["yhat_lower"][::-1]),
        fill="toself", fillcolor="rgba(232,89,12,0.15)", line=dict(color="rgba(0,0,0,0)"),
        name="Intervalo de confiança", showlegend=True,
    ))
    fig.update_layout(height=420, margin=dict(t=20, b=20), xaxis_title=None, yaxis_title="Incidentes/dia",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Datas críticas — próximos 7 dias")
    tabela = fc_futuro.head(7)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    tabela.columns = ["Data", "Previsão", "Mín. (80%)", "Máx. (80%)"]
    tabela["Acima da média histórica?"] = tabela["Previsão"] > media_hist
    for c in ["Previsão", "Mín. (80%)", "Máx. (80%)"]:
        tabela[c] = tabela[c].round(0).astype(int)
    st.dataframe(tabela, width="stretch", hide_index=True)

    if modo == "Técnico":
        with st.expander("Detalhes técnicos do modelo"):
            st.write(prophet_metrics)
            st.caption(
                "MAPE elevado em P2 reflete o crescimento acelerado da série (+382% Jul–Nov/2025) "
                "e o baixo volume absoluto em parte do histórico, que amplia o erro percentual. "
                "MAE (erro absoluto médio) é reportado como métrica complementar mais estável."
            )

# ------------------------------------------------------------------
# TELA 2 — Risco de OLA
# ------------------------------------------------------------------
elif tela.startswith("2"):
    st.title("Risco de Violação de OLA")
    st.caption("Modelo XGBoost | Universo: incidentes que entram na apuração de KPI")

    threshold = xgb_metrics["threshold"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Threshold do modelo", f"{threshold:.0%}")
    c2.metric("Precisão", f"{xgb_metrics['precision'].iloc[0]:.0%}")
    c3.metric("Recall", f"{xgb_metrics['recall'].iloc[0]:.0%}")
    c4.metric("ROC-AUC", f"{xgb_metrics['roc_auc'].iloc[0]:.2f}")

    alto_risco = scored[scored["risco_predito"] >= 0.70].sort_values("risco_predito", ascending=False)
    st.warning(f"⚠️ {len(alto_risco)} incidentes com risco de violação de OLA acima de 70%", icon="⚠️")

    st.subheader("Incidentes ordenados por risco")
    equipe_filtro = st.multiselect("Filtrar por equipe", sorted(scored["Grupo designado"].unique()))
    view = scored.sort_values("risco_predito", ascending=False)
    if equipe_filtro:
        view = view[view["Grupo designado"].isin(equipe_filtro)]
    view_show = view.head(200)[
        ["Número", "prioridade_label", "Grupo designado", "Categoria", "data_abertura", "risco_predito"]
    ].copy()
    view_show["risco_predito"] = (view_show["risco_predito"] * 100).round(1)
    view_show.columns = ["Incidente", "Prioridade", "Equipe", "Categoria", "Aberto em", "Risco (%)"]
    st.dataframe(
        view_show, width="stretch", hide_index=True,
        column_config={"Risco (%)": st.column_config.ProgressColumn("Risco (%)", min_value=0, max_value=100, format="%.1f%%")},
    )

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Resumo por equipe")
        resumo_eq = scored.groupby("Grupo designado").agg(
            incidentes=("Número", "count"), violacoes=("kpi_violado", "sum")
        ).reset_index()
        resumo_eq["taxa_violacao"] = (resumo_eq["violacoes"] / resumo_eq["incidentes"] * 100).round(1)
        resumo_eq = resumo_eq.sort_values("violacoes", ascending=False).head(10)
        fig_eq = px.bar(resumo_eq, x="Grupo designado", y="violacoes",
                         color="taxa_violacao", color_continuous_scale="Reds",
                         labels={"violacoes": "Violações de OLA", "Grupo designado": "Equipe", "taxa_violacao": "Taxa (%)"})
        fig_eq.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_eq, width="stretch")
    with colB:
        st.subheader("Histórico de violações no tempo")
        viol_tempo = scored[scored["kpi_violado"] == 1].copy()
        viol_tempo["mes"] = pd.to_datetime(viol_tempo["data_abertura"]).dt.to_period("M").astype(str)
        viol_mes = viol_tempo.groupby("mes").size().reset_index(name="violacoes")
        fig_tempo = px.line(viol_mes, x="mes", y="violacoes", markers=True)
        fig_tempo.update_layout(height=350, margin=dict(t=20, b=20), xaxis_title=None)
        st.plotly_chart(fig_tempo, width="stretch")

# ------------------------------------------------------------------
# TELA 3 — Análise Exploratória Interativa
# ------------------------------------------------------------------
elif tela.startswith("3"):
    st.title("Análise Exploratória Interativa")

    c1, c2, c3 = st.columns(3)
    prioridades = c1.multiselect("Prioridade", sorted(df["prioridade_label"].unique()),
                                  default=["P2", "P3"])
    produtos_top = df["Produto"].value_counts().head(15).index.tolist()
    produto_filtro = c2.multiselect("Produto (top 15)", produtos_top)
    periodo = c3.select_slider("Período (mês)", options=sorted(df["mes_abertura"].unique()),
                                value=(sorted(df["mes_abertura"].unique())[0], sorted(df["mes_abertura"].unique())[-1]))

    dff = df[df["prioridade_label"].isin(prioridades)] if prioridades else df
    dff = dff[(dff["mes_abertura"] >= periodo[0]) & (dff["mes_abertura"] <= periodo[1])]
    if produto_filtro:
        dff = dff[dff["Produto"].isin(produto_filtro)]

    st.subheader("Mapa de calor — volume por hora e dia da semana")
    ordem_dias = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
    heat = dff.groupby(["dia_semana", "hora_abertura"]).size().reset_index(name="incidentes")
    heat_pivot = heat.pivot(index="dia_semana", columns="hora_abertura", values="incidentes").reindex(ordem_dias)
    fig_heat = px.imshow(heat_pivot, color_continuous_scale="OrRd", aspect="auto",
                          labels=dict(x="Hora do dia", y="Dia da semana", color="Incidentes"))
    fig_heat.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig_heat, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ranking de produtos/categorias")
        top_produtos = dff["Produto"].value_counts().head(10).reset_index()
        top_produtos.columns = ["Produto", "Incidentes"]
        fig_prod = px.bar(top_produtos, x="Incidentes", y="Produto", orientation="h",
                           color="Incidentes", color_continuous_scale="Blues")
        fig_prod.update_layout(height=380, margin=dict(t=20, b=20), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_prod, width="stretch")
    with col2:
        st.subheader("Comparativo entre equipes ao longo do tempo")
        eq_top = dff["Grupo designado"].value_counts().head(6).index
        evol = dff[dff["Grupo designado"].isin(eq_top)].groupby(
            ["mes_abertura", "Grupo designado"]
        ).size().reset_index(name="incidentes")
        fig_evol = px.line(evol, x="mes_abertura", y="incidentes", color="Grupo designado")
        fig_evol.update_layout(height=380, margin=dict(t=20, b=20), xaxis_title=None)
        st.plotly_chart(fig_evol, width="stretch")

# ------------------------------------------------------------------
# TELA 4 — Recomendações Operacionais
# ------------------------------------------------------------------
else:
    st.title("Recomendações Operacionais")
    st.caption("Previsões traduzidas em ações preventivas para a operação")

    prox7 = forecast[forecast["ds"] > history["data_abertura"].max()].groupby("prioridade").head(7)
    p2_d7 = prox7[prox7["prioridade"] == "P2"]["yhat"].sum()
    p3_d7 = prox7[prox7["prioridade"] == "P3"]["yhat"].sum()
    equipes_risco = scored[scored["risco_predito"] >= 0.70]["Grupo designado"].value_counts().head(3)

    st.subheader("Painel executivo — próximos 7 dias")
    c1, c2, c3 = st.columns(3)
    c1.metric("Volume previsto P2", f"{p2_d7:.0f} incidentes")
    c2.metric("Volume previsto P3", f"{p3_d7:.0f} incidentes")
    c3.metric("Incidentes em alto risco de OLA", f"{len(scored[scored['risco_predito'] >= 0.70])}")

    st.markdown("#### Ações prioritárias recomendadas")
    recomendacoes = []
    if len(equipes_risco) > 0:
        recomendacoes.append(
            f"Reforçar plantão das equipes **{', '.join(equipes_risco.index)}** — concentram o maior número de incidentes em alto risco de violação de OLA."
        )
    recomendacoes.append("Priorizar triagem entre **09h–11h**, janela de maior volume histórico de incidentes.")
    recomendacoes.append(
        f"Preparar capacidade extra para o pico de **P3 previsto ({p3_d7:.0f} incidentes nos próximos 7 dias)**."
    )
    recomendacoes.append("Revisar causas recorrentes nas categorias com maior volume (ver Tela 3 — Análise Exploratória).")
    for r in recomendacoes:
        st.markdown(f"- {r}")

    st.markdown("#### Mapa de risco — próximos 7 dias por equipe e prioridade")
    risco_eq_prio = scored.groupby(["Grupo designado", "prioridade_label"])["risco_predito"].mean().reset_index()
    risco_eq_prio = risco_eq_prio[risco_eq_prio["Grupo designado"].isin(
        scored["Grupo designado"].value_counts().head(10).index
    )]
    pivot_risco = risco_eq_prio.pivot(index="Grupo designado", columns="prioridade_label", values="risco_predito").fillna(0)
    fig_risco = px.imshow(pivot_risco * 100, color_continuous_scale="Reds", text_auto=".1f",
                           labels=dict(x="Prioridade", y="Equipe", color="Risco médio (%)"))
    fig_risco.update_layout(height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig_risco, width="stretch")

    st.download_button(
        "📄 Exportar relatório (CSV)",
        data=scored.sort_values("risco_predito", ascending=False).head(50).to_csv(index=False),
        file_name="opspulse_recomendacoes.csv",
        mime="text/csv",
    )
    st.caption("Exportação em PDF prevista para a versão final (Sprint 4).")
