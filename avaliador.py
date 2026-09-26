import os
import csv
import glob
import time
import re
from collections import Counter
from llama_cpp import Llama
from rag_engine import RAGEngine

MODEL_PATH = os.path.join("modelo", "qwen2.5-3b-instruct-q4_k_m.gguf")
CSV_DIR = "fine-tuning"
OUTPUT_REPORT = os.path.join(CSV_DIR, "relatorio_avaliacao.csv")

MAX_PERGUNTAS = None

POSSIBLE_QUESTION_COLS = ["pergunta", "instruction", "input", "question", "prompt"]
POSSIBLE_ANSWER_COLS = ["resposta", "output", "expected_output", "target", "ground_truth"]


def normalizar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    return texto.split()


def calcular_rouge1(referencia, gerada):
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


def carregar_dados_csv(diretorio):
    arquivos_csv = glob.glob(os.path.join(diretorio, "*.csv"))
    # Filtra para evitar tentar ler o próprio relatório gerado caso ele já exista
    arquivos_csv = [f for f in arquivos_csv if os.path.basename(f) != "relatorio_avaliacao.csv"]
    
    if not arquivos_csv:
        print(f"[-] Nenhum arquivo CSV de dados encontrado na pasta '{diretorio}'.")
        return None

    caminho_csv = arquivos_csv[0]
    print(f"[+] Carregando dados de: {caminho_csv}")

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    delimitadores = [",", ";", "\t"]

    rows = []
    fieldnames = []

    for enc in encodings:
        for sep in delimitadores:
            try:
                with open(caminho_csv, mode="r", encoding=enc) as f:
                    reader = csv.DictReader(f, delimiter=sep)
                    temp_rows = list(reader)
                    if temp_rows and reader.fieldnames and len(reader.fieldnames) > 1:
                        rows = temp_rows
                        fieldnames = list(reader.fieldnames)
                        print(f"[+] CSV lido com sucesso (Encoding: {enc}, Delimitador: '{sep}')")
                        break
            except Exception:
                continue
        if rows:
            break

    if not rows:
        print("[-] Não foi possível ler o arquivo CSV com os encodings/delimitadores padrão.")
        return None

    return padronizar_dados(rows, fieldnames)


def padronizar_dados(rows, fieldnames):
    map_headers = {str(h).strip().lower(): h for h in fieldnames}

    col_pergunta = None
    for k in POSSIBLE_QUESTION_COLS:
        if k in map_headers:
            col_pergunta = map_headers[k]
            break

    col_resposta = None
    for k in POSSIBLE_ANSWER_COLS:
        if k in map_headers:
            col_resposta = map_headers[k]
            break

    if not col_pergunta:
        print(f"[-] Coluna de pergunta não encontrada. Colunas disponíveis: {fieldnames}")
        return None

    dados = []
    for r in rows:
        pergunta = r.get(col_pergunta, "").strip()
        resposta = r.get(col_resposta, "").strip() if col_resposta else ""

        ex1 = r.get("Exemplo 1", "").strip()
        ex2 = r.get("Exemplo 2", "").strip()
        ex3 = r.get("Exemplo 3", "").strip()

        if pergunta:
            dados.append({
                "Pergunta": pergunta,
                "Resposta_Esperada": resposta,
                "Exemplo_1": ex1,
                "Exemplo_2": ex2,
                "Exemplo_3": ex3
            })

    return dados


def inicializar_modelo(caminho_modelo):
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"[-] Modelo não encontrado em: {caminho_modelo}")

    num_threads = min(os.cpu_count() or 4, 6)

    print(f"[+] Carregando o modelo LLM de {caminho_modelo}...")
    print(f"[+] Threads CPU: {num_threads}")

    return Llama(
        model_path=caminho_modelo,
        n_ctx=2048,
        n_batch=512,
        n_threads=num_threads,
        n_gpu_layers=0,
        verbose=False
    )


def gerar_resposta(llm, prompt, max_tokens=128):
    response = llm(
        prompt,
        max_tokens=max_tokens,
        stop=["\nPergunta:", "\nUsuário:", "###", "Exemplo:"],
        echo=False,
        temperature=0.2
    )
    return response["choices"][0]["text"].strip()


def executar_avaliacao():
    dados = carregar_dados_csv(CSV_DIR)
    if not dados:
        print("[-] Nenhuma pergunta carregada. Verifique o arquivo CSV.")
        return

    if MAX_PERGUNTAS is not None:
        print(f"[!] MODO DE TESTE ATIVO: Processando apenas as primeiras {MAX_PERGUNTAS} perguntas.")
        dados = dados[:MAX_PERGUNTAS]

    print("\n[+] Inicializando RAG Engine para extração de documentos PDF...")
    rag = RAGEngine()
    tem_rag = rag.carregar_e_processar_pdfs()
    if not tem_rag:
        print("[!] Alerta: Nenhum PDF encontrado em 'documentos/'. A métrica 'Few-Shot + RAG' usará contexto vazio.")

    try:
        llm = inicializar_modelo(MODEL_PATH)
    except Exception as e:
        print(e)
        return

    resultados = []
    total = len(dados)
    tempo_inicio_total = time.time()

    scores_zero = []
    scores_one = []
    scores_few = []
    scores_few_rag = []

    print(f"\n[+] Iniciando avaliação de {total} perguntas em 4 estratégias...\n")

    for index, item in enumerate(dados, start=1):
        inicio_pergunta = time.time()
        pergunta = item["Pergunta"]
        esperada = item["Resposta_Esperada"]

        ex_one_shot = f"Exemplo:\n{item['Exemplo_1']}\n\n" if item["Exemplo_1"] else ""

        ex_few_shot = ""
        for k in ["Exemplo_1", "Exemplo_2", "Exemplo_3"]:
            if item[k]:
                ex_few_shot += f"- {item[k]}\n"
        ex_few_shot_str = f"Exemplos de contexto:\n{ex_few_shot}\n" if ex_few_shot else ""

        contexto_rag, _ = rag.buscar_contexto(pergunta) if tem_rag else ("", [])

        prompt_zero = f"Você é um especialista em agronomia. Responda de forma objetiva.\n\nPergunta: {pergunta}\nResposta:"
        prompt_one = f"Você é um especialista em agronomia.\n\n{ex_one_shot}Pergunta: {pergunta}\nResposta:"
        prompt_few = f"Você é um especialista em agronomia.\n\n{ex_few_shot_str}Pergunta: {pergunta}\nResposta:"
        prompt_few_rag = (
            f"Você é um especialista em agronomia.\n\n"
            f"Documentos de referência:\n{contexto_rag}\n\n"
            f"{ex_few_shot_str}"
            f"Pergunta: {pergunta}\nResposta:"
        )

        print(f" -> [{index}/{total}] Processando respostas...", end="", flush=True)

        resp_zero = gerar_resposta(llm, prompt_zero)
        resp_one = gerar_resposta(llm, prompt_one)
        resp_few = gerar_resposta(llm, prompt_few)
        resp_few_rag = gerar_resposta(llm, prompt_few_rag)

        score_zero = calcular_rouge1(esperada, resp_zero)
        score_one = calcular_rouge1(esperada, resp_one)
        score_few = calcular_rouge1(esperada, resp_few)
        score_few_rag = calcular_rouge1(esperada, resp_few_rag)

        scores_zero.append(score_zero)
        scores_one.append(score_one)
        scores_few.append(score_few)
        scores_few_rag.append(score_few_rag)

        tempo_gasto = time.time() - inicio_pergunta
        print(f" Concluído em {tempo_gasto:.1f}s")

        resultados.append({
            "Indice": index,
            "Pergunta": pergunta,
            "Resposta_Esperada": esperada,
            "Zero_Shot": resp_zero,
            "ROUGE1_Zero_Shot": f"{score_zero}%",
            "One_Shot": resp_one,
            "ROUGE1_One_Shot": f"{score_one}%",
            "Few_Shot": resp_few,
            "ROUGE1_Few_Shot": f"{score_few}%",
            "Few_Shot_RAG": resp_few_rag,
            "ROUGE1_Few_Shot_RAG": f"{score_few_rag}%"
        })

    tempo_total = time.time() - tempo_inicio_total

    media_zero = round(sum(scores_zero) / len(scores_zero), 2) if scores_zero else 0
    media_one = round(sum(scores_one) / len(scores_one), 2) if scores_one else 0
    media_few = round(sum(scores_few) / len(scores_few), 2) if scores_few else 0
    media_few_rag = round(sum(scores_few_rag) / len(scores_few_rag), 2) if scores_few_rag else 0

    print(f"\n[+] Processamento completo concluído em {tempo_total/60:.2f} minutos.")
    print("\n=== RESUMO DAS MÉTRICAS MÉDIAS (ROUGE-1 F1) ===")
    print(f" -> Média Zero-Shot:     {media_zero}%")
    print(f" -> Média One-Shot:      {media_one}%")
    print(f" -> Média Few-Shot:      {media_few}%")
    print(f" -> Média Few-Shot + RAG: {media_few_rag}%")
    print("================================================\n")

    salvar_relatorio(resultados, OUTPUT_REPORT)


def salvar_relatorio(resultados, caminho_saida):
    if not resultados:
        return

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    fieldnames = [
        "Indice",
        "Pergunta",
        "Resposta_Esperada",
        "Zero_Shot",
        "ROUGE1_Zero_Shot",
        "One_Shot",
        "ROUGE1_One_Shot",
        "Few_Shot",
        "ROUGE1_Few_Shot",
        "Few_Shot_RAG",
        "ROUGE1_Few_Shot_RAG"
    ]

    with open(caminho_saida, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(resultados)

    print(f"[+] Relatório salvo em: {caminho_saida}")


if __name__ == "__main__":
    executar_avaliacao()