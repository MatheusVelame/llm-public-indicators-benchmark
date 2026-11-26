import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("results/figures", exist_ok=True)

def run_analysis(csv_path="results/evaluated_results.csv"):
    df = pd.read_csv(csv_path)

    # 1. Tabela resumo por Modelo e Estratégia
    summary = df.groupby(["model", "strategy"]).agg(
        accuracy=("is_accurate", "mean"),
        hallucination_rate=("hallucination", "mean"),
        citation_hallucination_rate=("citation_hallucination", "mean"),
        median_era=("era", "median"),
        avg_latency_s=("latency_s", "mean")
    ).reset_index()

    # Calibração de incerteza (apenas perguntas irrespondíveis)
    unans = df[df["answerable"] == False].groupby(["model", "strategy"])["calibrated_uncertainty"].mean().reset_index()
    summary = summary.merge(unans, on=["model", "strategy"], how="left")

    summary.to_csv("results/summary_table.csv", index=False)
    print("\n=== TABELA RESUMO ===")
    print(summary.to_string(index=False))

    # 2. Gráfico: Acurácia vs Taxa de Alucinação por Estratégia
    sns.set_theme(style="whitegrid")
    melted = summary.melt(
        id_vars=["model", "strategy"],
        value_vars=["accuracy", "hallucination_rate", "citation_hallucination_rate"],
        var_name="Métrica", value_name="Taxa"
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=melted, x="strategy", y="Taxa", hue="Métrica")
    plt.title("Comparação de Confiabilidade: Zero-Shot vs RAG vs Tool Use")
    plt.ylim(0, 1.05)
    plt.ylabel("Proporção (0 a 1)")
    plt.xlabel("Estratégia de Acesso")
    plt.tight_layout()
    plt.savefig("results/figures/reliability_comparison.png", dpi=300)
    plt.close()

    # 3. Gráfico: Latência Média por Estratégia
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="strategy", y="latency_s")
    plt.title("Distribuição de Latência por Estratégia")
    plt.ylabel("Latência (segundos)")
    plt.xlabel("Estratégia")
    plt.tight_layout()
    plt.savefig("results/figures/latency_comparison.png", dpi=300)
    plt.close()

    print("\n[OK] Gráficos salvos em results/figures/.")

if __name__ == "__main__":
    run_analysis()