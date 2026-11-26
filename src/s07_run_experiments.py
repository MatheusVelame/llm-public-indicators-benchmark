import os
import json
from tqdm import tqdm
from src.s04_zeroshot import run_zeroshot
from src.s05_rag import RAGPipeline
from src.s06_tool_use import run_tool_use

os.makedirs("results", exist_ok=True)

# Configure aqui os modelos que deseja avaliar
MODELS_CONFIG = [
    {"provider": "openai", "model": "gpt-4o-mini"},
    # {"provider": "gemini", "model": "gemini-2.0-flash"},
    # {"provider": "groq", "model": "llama-3.3-70b-versatile"},
]

def run_all(benchmark_path="data/benchmark.json", output_path="results/raw_outputs.jsonl", limit=None):
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    if limit:
        benchmark = benchmark[:limit]

    rag = RAGPipeline()

    with open(output_path, "w", encoding="utf-8") as out_f:
        for cfg in MODELS_CONFIG:
            provider, model = cfg["provider"], cfg["model"]
            print(f"\n=== Executando modelo: {model} ({provider}) ===")

            for item in tqdm(benchmark, desc=model):
                q = item["question"]
                strategies = {
                    "zero_shot": lambda: run_zeroshot(q, model=model, provider=provider),
                    "rag": lambda: rag.query(q, model=model, provider=provider),
                    "tool_use": lambda: run_tool_use(q, model=model, provider=provider),
                }

                for strat_name, fn in strategies.items():
                    try:
                        res = fn()
                    except Exception as e:
                        res = {"answer": f"ERROR: {e}", "latency_s": 0.0, "steps": []}

                    record = {
                        "id": item["id"],
                        "category": item["category"],
                        "answerable": item["answerable"],
                        "ground_truth": item["ground_truth"],
                        "model": model,
                        "strategy": strat_name,
                        "response": res["answer"],
                        "latency_s": res["latency_s"],
                        "steps": res["steps"]
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()

    print(f"[OK] Experimentos finalizados. Logs salvos em {output_path}.")

if __name__ == "__main__":
    # Use limit=5 para um teste rápido antes de rodar o benchmark completo
    run_all(limit=None)