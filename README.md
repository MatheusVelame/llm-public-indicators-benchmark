# Confiabilidade de Modelos de Linguagem na Consulta a Indicadores Públicos

Repositório oficial do benchmark experimental desenvolvido para o Pré Trabalho de Conclusão de Curso (Pré-TCC) em Ciência da Computação — **CESAR School**.

**Autores:** Matheus Velame Pessoa & Lucas Gallindo Martins Soares  
**Ano:** 2026

---

## Sobre o Projeto

Este projeto avalia empiricamente a confiabilidade de Modelos de Linguagem de Grande Porte (LLMs) na consulta a indicadores públicos socioeconômicos e de governança (World Bank / WDI / WGI), comparando diretamente três estratégias de acesso a dados:

1. **Zero-Shot (Conhecimento Paramétrico):** O modelo responde sem acesso a fontes externas.
2. **RAG (Retrieval-Augmented Generation):** O modelo consulta relatórios textuais indexados em um banco vetorial local (ChromaDB + SentenceTransformers).
3. **Tool Use (SQL / Function Calling):** O modelo gera e executa consultas SQL em modo somente leitura diretamente sobre o banco relacional (`SQLite`) dos indicadores.

### Métricas Unificadas
- **Acurácia (`accuracy`):** Proporção de respostas numéricas dentro da tolerância de erro relativo ($\le 5\%$) ou abstenção correta em perguntas irrespondíveis.
- **Erro Relativo Absoluto Mediano (`median_era`):** Precisão quantitativa das respostas extraídas frente ao gabarito determinístico.
- **Taxa de Alucinação (`hallucination_rate`):** Respostas incorretas ou inventadas sem expressão de incerteza.
- **Alucinação de Citação (`citation_hallucination_rate`):** Casos em que o modelo alucina um valor e atribui falsamente a estatística a uma fonte oficial (ex.: Banco Mundial).
- **Calibração de Incerteza (`calibrated_uncertainty`):** Capacidade do modelo de admitir desconhecimento em perguntas fora do recorte temporal (ex.: dados futuros).

---

## Resultados Experimentais (`openai/gpt-oss-120b`)

Avaliação executada sobre 65 instâncias de teste (valor exato, comparação temporal, agregação e perguntas irrespondíveis), totalizando 195 execuções:

| Modelo | Estratégia | Acurácia | Taxa de Alucinação | Alucinação de Citação | ERA Mediano | Latência Média (s) | Calibração de Incerteza |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `openai/gpt-oss-120b` | **Zero-Shot** | 47.69% | 20.00% | 9.23% | 0.0212 | 2.08s | 100.0% |
| `openai/gpt-oss-120b` | **Tool Use (SQL)** | 96.92% | 3.08% | 0.00% | 0.0000 | 7.04s | 100.0% |
| `openai/gpt-oss-120b` | **RAG** | 100.00% | 0.00% | 0.00% | 0.0000 | 7.95s | 100.0% |

Os gráficos comparativos gerados automaticamente estão disponíveis em `results/figures/`:
- `results/figures/reliability_comparison.png`
- `results/figures/latency_comparison.png`

---

## Estrutura do Repositório

```text
llm-public-indicators-benchmark/
├── .env.example                 # Template de variáveis de ambiente (chaves de API)
├── requirements.txt             # Dependências Python
├── data/
│   ├── indicators.db            # Base relacional SQLite (World Bank)
│   ├── indicators.csv           # Exportação tabular dos indicadores
│   ├── benchmark.json           # Conjunto de perguntas e gabaritos determinísticos
│   └── docs/                    # Relatórios textuais por país para indexação RAG
├── results/
│   ├── evaluated_results.csv    # Métricas detalhadas por instância
│   ├── summary_table.csv        # Tabela resumo consolidada
│   └── figures/                 # Gráficos de confiabilidade e latência
└── src/
    ├── llm_client.py            # Cliente unificado (OpenAI, Gemini, Groq)
    ├── s01_data_curation.py     # Coleta via API do World Bank (SQLite + Docs)
    ├── s02_generate_benchmark.py# Geração automatizada do benchmark e gabaritos
    ├── s03_metrics.py           # Implementação das métricas unificadas
    ├── s04_zeroshot.py          # Pipeline Zero-Shot
    ├── s05_rag.py               # Pipeline RAG (ChromaDB)
    ├── s06_tool_use.py          # Pipeline Tool Use (SQLite Function Calling)
    ├── s07_run_experiments.py   # Executor controlado dos experimentos
    ├── s08_evaluate.py          # Processamento das métricas sobre os logs
    └── s09_analysis.py          # Análise estatística e geração de gráficos