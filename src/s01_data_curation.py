import os
import sqlite3
import requests
import pandas as pd

os.makedirs("data/docs", exist_ok=True)

# Indicadores do World Bank (WDI / Governança)
INDICATORS = {
    "NY.GDP.MKTP.CD": "PIB (USD correntes)",
    "SP.POP.TOTL": "População total",
    "EG.ELC.ACCS.ZS": "Acesso à eletricidade (% da população)",
    "EN.ATM.CO2E.PC": "Emissões de CO2 (toneladas métricas per capita)",
    "CC.EST": "Controle da Corrupção (WGI Estimate)"
}

COUNTRIES = ["BRA", "USA", "CHE", "NGA", "IND", "ZAF", "DEU", "JPN", "ARG", "KEN"]
YEARS = "2018:2023"

def fetch_world_bank_data():
    records = []
    for ind_code, ind_name in INDICATORS.items():
        url = f"https://api.worldbank.org/v2/country/{';'.join(COUNTRIES)}/indicator/{ind_code}?date={YEARS}&format=json&per_page=1000"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if len(data) < 2 or not data[1]:
            continue
        for entry in data[1]:
            if entry["value"] is not None:
                records.append({
                    "country_code": entry["countryiso3code"],
                    "country_name": entry["country"]["value"],
                    "indicator_code": ind_code,
                    "indicator_name": ind_name,
                    "year": int(entry["date"]),
                    "value": float(entry["value"])
                })
    return pd.DataFrame(records)

def save_sqlite(df: pd.DataFrame, db_path="data/indicators.db"):
    conn = sqlite3.connect(db_path)
    df.to_sql("indicators", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[OK] Banco SQLite salvo em {db_path} ({len(df)} registros).")

def generate_text_documents(df: pd.DataFrame, docs_dir="data/docs"):
    """Gera documentos textuais estruturados simulando relatórios institucionais para o RAG."""
    for country, group in df.groupby("country_name"):
        filename = os.path.join(docs_dir, f"{country.replace(' ', '_')}_report.txt")
        lines = [f"Relatório de Indicadores Públicos - Banco Mundial (World Bank)\nPaís: {country}\n"]
        for ind_name, ind_group in group.groupby("indicator_name"):
            lines.append(f"\nIndicador: {ind_name}")
            for _, row in ind_group.sort_values("year").iterrows():
                lines.append(f"No ano de {row['year']}, o indicador {ind_name} para {country} registrou o valor de {row['value']:.4f}.")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    print(f"[OK] Documentos para RAG gerados em {docs_dir}/.")

if __name__ == "__main__":
    print("Coletando dados da API do World Bank...")
    df = fetch_world_bank_data()
    df.to_csv("data/indicators.csv", index=False)
    save_sqlite(df)
    generate_text_documents(df)