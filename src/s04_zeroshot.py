import time
from src.llm_client import get_client

SYSTEM_PROMPT = (
    "Você é um assistente de análise de indicadores públicos. "
    "Responda de forma objetiva. Se souber o valor, termine sua resposta com 'VALOR: <numero>'. "
    "Se o dado não existir, for futuro ou você não tiver certeza, diga explicitamente que não sabe."
)

def run_zeroshot(question: str, model: str = "gpt-4o-mini", provider: str = "openai") -> dict:
    client = get_client(provider)
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
    )
    latency = time.perf_counter() - start
    return {
        "answer": resp.choices[0].message.content,
        "latency_s": round(latency, 3),
        "steps": []
    }