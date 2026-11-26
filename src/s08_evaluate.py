import json
import pandas as pd
from src.s03_metrics import evaluate_response

def process_results(raw_path="results/raw_outputs.jsonl", out_csv="results/evaluated_results.csv"):
    rows = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            metrics = evaluate_response(
                response_text=rec["response"],
                ground_truth=rec["ground_truth"],
                answerable=rec["answerable"]
            )
            rows.append({
                "id": rec["id"],
                "category": rec["category"],
                "answerable": rec["answerable"],
                "model": rec["model"],
                "strategy": rec["strategy"],
                "ground_truth": rec["ground_truth"],
                "latency_s": rec["latency_s"],
                **metrics
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Resultados avaliados salvos em {out_csv} ({len(df)} linhas).")

if __name__ == "__main__":
    process_results()