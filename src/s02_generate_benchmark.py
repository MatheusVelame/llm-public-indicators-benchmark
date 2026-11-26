import json
import sqlite3
import pandas as pd

def generate_benchmark(db_path="data/indicators.db", out_path="data/benchmark.json"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM indicators", conn)
    conn.close()

    benchmark = []
    qid = 1

    # 1. Recuperação de Valor Exato (Factoid)
    sample_exact = df.sample(min(30, len(df)), random_state=42)
    for _, row in sample_exact.iterrows():
        benchmark.append({
            "id": f"Q{qid:03d}",
            "category": "exact_value",
            "answerable": True,
            "question": f"Qual foi o valor do indicador '{row['indicator_name']}' ({row['indicator_code']}) para o país {row['country_name']} no ano de {row['year']} segundo o Banco Mundial?",
            "ground_truth": row["value"],
            "metadata": {"country": row["country_name"], "indicator": row["indicator_code"], "year": int(row["year"])}
        })
        qid += 1

    # 2. Comparação Temporal (Diferença entre dois anos)
    for (country, ind_code, ind_name), group in df.groupby(["country_name", "indicator_code", "indicator_name"]):
        if len(group) >= 2 and qid <= 45:
            g = group.sort_values("year")
            y1, v1 = int(g.iloc[0]["year"]), float(g.iloc[0]["value"])
            y2, v2 = int(g.iloc[-1]["year"]), float(g.iloc[-1]["value"])
            benchmark.append({
                "id": f"Q{qid:03d}",
                "category": "temporal_comparison",
                "answerable": True,
                "question": f"Qual foi a diferença absoluta (valor de {y2} menos o valor de {y1}) do indicador '{ind_name}' para {country}?",
                "ground_truth": round(v2 - v1, 4),
                "metadata": {"country": country, "indicator": ind_code, "years": [y1, y2]}
            })
            qid += 1

    # 3. Agregação (Média histórica no período)
    for (country, ind_code, ind_name), group in df.groupby(["country_name", "indicator_code", "indicator_name"]):
        if len(group) >= 3 and qid <= 60:
            years = sorted(group["year"].tolist())
            benchmark.append({
                "id": f"Q{qid:03d}",
                "category": "aggregation_mean",
                "answerable": True,
                "question": f"Qual foi a média do indicador '{ind_name}' para {country} considerando os dados disponíveis entre {years[0]} e {years[-1]}?",
                "ground_truth": round(float(group["value"].mean()), 4),
                "metadata": {"country": country, "indicator": ind_code}
            })
            qid += 1

    # 4. Perguntas Irrespondíveis / Fora do Recorte (Calibração de Incerteza)
    unanswerable_countries = ["Brasil", "Suíça", "Nigéria", "Índia", "Japão"]
    for country in unanswerable_countries:
        benchmark.append({
            "id": f"Q{qid:03d}",
            "category": "unanswerable_future",
            "answerable": False,
            "question": f"Qual foi o valor oficial consolidado do PIB (NY.GDP.MKTP.CD) para {country} no ano de 2029 segundo o Banco Mundial?",
            "ground_truth": None,
            "metadata": {"country": country, "year": 2029}
        })
        qid += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
    print(f"[OK] Benchmark gerado com {len(benchmark)} perguntas em {out_path}.")

if __name__ == "__main__":
    generate_benchmark()