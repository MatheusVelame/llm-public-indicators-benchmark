import json
import time
import sqlite3
from src.llm_client import get_client

DB_PATH = "data/indicators.db"

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_sql",
            "description": (
                "Executa uma consulta SELECT no banco SQLite de indicadores públicos. "
                "Tabela: indicators(country_code TEXT, country_name TEXT, indicator_code TEXT, "
                "indicator_name TEXT, year INTEGER, value REAL)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "Consulta SQL SELECT válida."}
                },
                "required": ["sql"]
            }
        }
    }
]

def execute_safe_sql(sql: str) -> str:
    if not sql.strip().lower().startswith("select"):
        return "ERRO: Apenas consultas SELECT são permitidas."
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        conn.close()
        return json.dumps({"columns": cols, "rows": rows[:50]}, ensure_ascii=False)
    except Exception as e:
        return f"ERRO SQL: {str(e)}"

def run_tool_use(question: str, model: str = "gpt-4o-mini", provider: str = "openai") -> dict:
    client = get_client(provider)
    start = time.perf_counter()
    steps = []

    messages = [
        {
            "role": "system",
            "content": (
                "Você tem acesso ao banco SQLite de indicadores públicos via ferramenta query_sql. "
                "Sempre consulte o banco antes de responder. Se a consulta não retornar dados, "
                "diga explicitamente que não há dados disponíveis. Ao responder com número, termine com 'VALOR: <numero>'."
            )
        },
        {"role": "user", "content": question}
    ]

    for _ in range(3):  # Máximo de 3 iterações de tool calling
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto"
        )
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            latency = time.perf_counter() - start
            return {"answer": msg.content or "", "latency_s": round(latency, 3), "steps": steps}

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            sql_result = execute_safe_sql(args.get("sql", ""))
            steps.append({"tool": tc.function.name, "sql": args.get("sql"), "output": sql_result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": sql_result
            })

    latency = time.perf_counter() - start
    return {"answer": "Não foi possível concluir a consulta.", "latency_s": round(latency, 3), "steps": steps}