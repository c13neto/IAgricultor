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

def avaliar_llm_judge(llm, pergunta: str, esperada: str, gerada: str) -> float:
    prompt_judge = (
        f"Você é um avaliador rigoroso e imparcial.\n"
        f"Pergunta original: {pergunta}\n"
        f"Resposta de gabarito: {esperada}\n"
        f"Resposta gerada pela IA: {gerada}\n\n"
        f"A resposta gerada possui o mesmo significado e atende ao gabarito corretamente, mesmo usando palavras diferentes? "
        f"Avalie e forneça uma nota entre 0.0 e 1.0 (onde 1.0 significa acerto total no significado e 0.0 erro total). "
        f"Responda APENAS com o número decimal. Nada mais."
    )
    
    try:
        nota_str = llm.generate_text(prompt_judge, max_tokens=10, temperature=0.1).strip()
        numeros = re.findall(r"0\.\d+|1\.0|0|1", nota_str)
        if numeros:
            return round(float(numeros[0]) * 100.0, 2)
        return 0.0
    except Exception:
        return 0.0
