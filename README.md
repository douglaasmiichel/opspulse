# OpsPulse — AIOps para Previsão de Incidentes de TI

**Locaweb Challenge 2026 | Turma 2TSCOA | Grupo: Os Mosqueteiros**
Beatriz Cassemiro Costa (RM566271) · Douglas Michel da Silva Costa (RM564209)

MVP preliminar — Sprint 3.

## O que está aqui

Implementação das 4 camadas da arquitetura definida na Sprint 2:

| Camada | Arquivo | Descrição |
|---|---|---|
| 1. Ingestão | `data/LW-DATASET.xlsx` | 122.543 incidentes reais da Locaweb (2023–2025) |
| 2. Processamento | `data_prep.py` | Limpeza, tratamento de nulos, features temporais |
| 3. Modelagem — Prophet | `train_prophet.py` | Previsão de volume D+1/D+7 para P2 e P3 |
| 3. Modelagem — XGBoost | `train_xgboost.py` | Classificador de risco de violação de OLA |
| 4. Visualização | `app.py` | Dashboard Streamlit + Plotly, 4 telas |

## Como rodar

```bash
pip install -r requirements.txt
python data_prep.py          # valida a limpeza dos dados
python train_prophet.py      # treina Prophet e gera models/prophet_forecast.csv
python train_xgboost.py      # treina XGBoost e gera models/xgboost_scored.csv
streamlit run app.py         # abre o dashboard em localhost:8501
```

## Resultados obtidos (validação)

**Prophet (previsão de volume, holdout de 14 dias):**
- P3: MAPE 50,8% | MAE 231,7 incidentes/dia (608 dias de histórico)
- P2: MAPE 161,2% | MAE 49,7 incidentes/dia (367 dias de histórico)
  - MAPE de P2 é inflado pelo crescimento acelerado da série (+382% jul–nov/2025) e por dias de baixo volume absoluto (ex.: feriados de fim de ano), que ampliam o erro percentual. MAE é reportado como métrica complementar mais estável.

**XGBoost (risco de violação de OLA, classe minoritária ~1%):**
- Threshold otimizado: 0,85 | F1: 0,17 | Precisão: 0,14 | Recall: 0,21 | ROC-AUC: 0,77
- Principais features: equipe designada, dia útil, categoria, hora de abertura

## Próximos passos (Sprint 4)

- Tuning adicional dos hiperparâmetros (Prophet: regressors externos; XGBoost: grid search)
- Exportação de relatório em PDF na Tela 4
- Deploy em produção no Streamlit Cloud
