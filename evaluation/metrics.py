import re
from collections import Counter

def normalizar_texto(texto: str) -> list[str]:
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    return texto.split()

def calcular_rouge1(referencia: str, gerada: str) -> float:
    tokens_ref = normalizar_texto(referencia)
    tokens_ger = normalizar_texto(gerada)

    if not tokens_ref or not tokens_ger:
        return 0.0

    intersecao = sum((Counter(tokens_ref) & Counter(tokens_ger)).values())

    precision = intersecao / len(tokens_ger)
    recall = intersecao / len(tokens_ref)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1 * 100, 2)
