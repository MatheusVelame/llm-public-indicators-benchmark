import re
from typing import Optional, Dict, Any

UNCERTAINTY_PATTERNS = [
    r"não (sei|tenho|há|existem|foi possível|possuo)",
    r"desconhecid[oa]",
    r"fora do (escopo|recorte|período)",
    r"dados (indisponíveis|não disponíveis|inexistentes|futuros)",
    r"impossível (saber|determinar|calcular)",
    r"cannot (answer|determine|provide)",
    r"no data available"
]

CITATION_PATTERNS = [
    r"banco mundial",
    r"world bank",
    r"segundo os dados",
    r"de acordo com",
    r"fonte:"
]

def extract_first_number(text: str) -> Optional[float]:
    """Extrai o valor numérico principal da resposta do LLM."""
    if not text:
        return None
    # Procura por marcação explícita primeiro (ex: VALOR: 1234.56)
    explicit = re.search(r"VALOR:\s*([-+]?\d[\d.,]*(?:[eE][-+]?\d+)?)", text, re.IGNORECASE)
    target = explicit.group(1) if explicit else None

    if not target:
        matches = re.findall(r"[-+]?\d[\d.,]*", text)
        if not matches:
            return None
        target = matches[-1]  # Geralmente o resultado final aparece por último

    # Normaliza separadores (ex: 1.234,56 ou 1,234.56)
    if "," in target and "." in target:
        if target.rfind(",") > target.rfind("."):
            target = target.replace(".", "").replace(",", ".")
        else:
            target = target.replace(",", "")
    elif "," in target:
        target = target.replace(",", ".")

    try:
        return float(target)
    except ValueError:
        return None

def expresses_uncertainty(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in UNCERTAINTY_PATTERNS)

def has_official_citation(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in CITATION_PATTERNS)

def evaluate_response(response_text: str, ground_truth: Optional[float], answerable: bool, tol: float = 0.05) -> Dict[str, Any]:
    """
    Calcula métricas unificadas:
    - absolute_relative_error (ERA)
    - is_accurate (ERA <= tol)
    - hallucination (inventou dado em pergunta irrespondível ou errou acima da tolerância sem admitir incerteza)
    - citation_hallucination (alucinou e ainda citou fonte oficial)
    - calibrated_uncertainty (admitiu incerteza quando answerable=False)
    """
    uncertain = expresses_uncertainty(response_text)
    cited = has_official_citation(response_text)
    extracted = extract_first_number(response_text)

    if not answerable:
        hallucinated = (extracted is not None) and (not uncertain)
        return {
            "extracted_value": extracted,
            "era": None,
            "is_accurate": uncertain and not hallucinated,
            "expressed_uncertainty": uncertain,
            "hallucination": hallucinated,
            "citation_hallucination": hallucinated and cited,
            "calibrated_uncertainty": uncertain
        }

    # Pergunta respondível
    if extracted is None or ground_truth is None:
        return {
            "extracted_value": None,
            "era": None,
            "is_accurate": False,
            "expressed_uncertainty": uncertain,
            "hallucination": not uncertain,
            "citation_hallucination": (not uncertain) and cited,
            "calibrated_uncertainty": False
        }

    denom = abs(ground_truth) if abs(ground_truth) > 1e-9 else 1e-9
    era = abs(extracted - ground_truth) / denom
    is_accurate = era <= tol
    hallucinated = (not is_accurate) and (not uncertain)

    return {
        "extracted_value": extracted,
        "era": round(era, 6),
        "is_accurate": is_accurate,
        "expressed_uncertainty": uncertain,
        "hallucination": hallucinated,
        "citation_hallucination": hallucinated and cited,
        "calibrated_uncertainty": False
    }