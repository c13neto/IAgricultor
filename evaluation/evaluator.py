import os
import csv
import glob
from typing import List, Dict, Any
from core.llm_service import LLMService
from core.rag_engine import RAGEngine
from evaluation.metrics import avaliar_llm_judge
from config import DATASET_DIR

class PromptFactory:
    @staticmethod
    def create_zero_shot(pergunta: str) -> str:
        return f"Você é um especialista em agronomia. Responda de forma objetiva.\n\nPergunta: {pergunta}\nResposta:"
        
    @staticmethod
    def create_one_shot(pergunta: str, exemplo: str) -> str:
        ex = f"Exemplo:\n{exemplo}\n\n" if exemplo else ""
        return f"Você é um especialista em agronomia.\n\n{ex}Pergunta: {pergunta}\nResposta:"
        
    @staticmethod
    def create_few_shot(pergunta: str, exemplos: List[str]) -> str:
        ex_str = ""
        valid_examples = [e for e in exemplos if e]
        if valid_examples:
            ex_str = "Exemplos de contexto:\n" + "\n".join([f"- {e}" for e in valid_examples]) + "\n\n"
        return f"Você é um especialista em agronomia.\n\n{ex_str}Pergunta: {pergunta}\nResposta:"
        
    @staticmethod
    def create_rag(pergunta: str, exemplos: List[str], contexto_rag: str) -> str:
        ex_str = ""
        valid_examples = [e for e in exemplos if e]
        if valid_examples:
            ex_str = "Exemplos de contexto:\n" + "\n".join([f"- {e}" for e in valid_examples]) + "\n\n"
            
        return (
            f"Você é um especialista em agronomia.\n\n"
            f"Sua tarefa é responder a seguinte pergunta baseando-se nas referências abaixo.\n"
            f"Pergunta a ser respondida: {pergunta}\n\n"
            f"Documentos de referência:\n{contexto_rag}\n\n"
            f"{ex_str}"
            f"Lembre-se de focar diretamente na Pergunta: {pergunta}\nResposta:"
        )

class Evaluator:
    POSSIBLE_QUESTION_COLS = ["pergunta", "instruction", "input", "question", "prompt"]
    POSSIBLE_ANSWER_COLS = ["resposta", "output", "expected_output", "target", "ground_truth"]
    
    def __init__(self, llm_service: LLMService, rag_engine: RAGEngine):
        self.llm = llm_service
        self.rag = rag_engine
        
    def _carregar_dados_csv(self) -> List[Dict[str, Any]]:
        arquivos_csv = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
        arquivos_csv = [f for f in arquivos_csv if os.path.basename(f) != "relatorio_avaliacao.csv"]
        
        if not arquivos_csv:
            print(f"[-] Nenhum arquivo CSV encontrado em '{DATASET_DIR}'.")
            return []
            
        caminho_csv = arquivos_csv[0]
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
                            break
                except Exception:
                    continue
            if rows:
                break
                
        if not rows:
            return []
            
        map_headers = {str(h).strip().lower(): h for h in fieldnames}
        col_pergunta = next((map_headers[k] for k in self.POSSIBLE_QUESTION_COLS if k in map_headers), None)
        col_resposta = next((map_headers[k] for k in self.POSSIBLE_ANSWER_COLS if k in map_headers), None)
        
        if not col_pergunta:
            return []
            
        dados = []
        for r in rows:
            if p := r.get(col_pergunta, "").strip():
                dados.append({
                    "Pergunta": p,
                    "Resposta_Esperada": r.get(col_resposta, "").strip() if col_resposta else "",
                    "Exemplo_1": r.get("Exemplo 1", "").strip(),
                    "Exemplo_2": r.get("Exemplo 2", "").strip(),
                    "Exemplo_3": r.get("Exemplo 3", "").strip()
                })
        return dados

    def run_benchmark(self, max_perguntas: int | None = None):
        dados = self._carregar_dados_csv()
        if not dados:
            print("[-] Nenhuma pergunta carregada.")
            return
            
        if max_perguntas:
            dados = dados[:max_perguntas]
            
        tem_rag = self.rag.carregar_e_processar_pdfs()
        
        resultados = []
        scores = {"zero": [], "one": [], "few": [], "few_rag": []}
        
        print(f"\n[+] Iniciando avaliação de {len(dados)} perguntas...\n")
        
        for idx, item in enumerate(dados, start=1):
            pergunta = item["Pergunta"]
            esperada = item["Resposta_Esperada"]
            exemplos = [item["Exemplo_1"], item["Exemplo_2"], item["Exemplo_3"]]
            
            contexto_rag, _ = self.rag.buscar_contexto(pergunta) if tem_rag else ("", [])
            
            p_zero = PromptFactory.create_zero_shot(pergunta)
            p_one = PromptFactory.create_one_shot(pergunta, item["Exemplo_1"])
            p_few = PromptFactory.create_few_shot(pergunta, exemplos)
            p_rag = PromptFactory.create_rag(pergunta, exemplos, contexto_rag)
            
            print(f" -> [{idx}/{len(dados)}] Processando pergunta...")
            
            try:
                r_zero = self.llm.generate_text(p_zero)
                r_one = self.llm.generate_text(p_one)
                r_few = self.llm.generate_text(p_few)
                r_rag = self.llm.generate_text(p_rag)
            except Exception as e:
                print(f"[-] Erro ao inferir pergunta {idx}: {e}")
                continue
                
            s_zero = avaliar_llm_judge(self.llm, pergunta, esperada, r_zero)
            s_one = avaliar_llm_judge(self.llm, pergunta, esperada, r_one)
            s_few = avaliar_llm_judge(self.llm, pergunta, esperada, r_few)
            s_rag = avaliar_llm_judge(self.llm, pergunta, esperada, r_rag)
            
            scores["zero"].append(s_zero)
            scores["one"].append(s_one)
            scores["few"].append(s_few)
            scores["few_rag"].append(s_rag)
            
            resultados.append({
                "Indice": idx,
                "Pergunta": pergunta,
                "Resposta_Esperada": esperada,
                "Zero_Shot": r_zero,
                "LLM_Judge_Zero_Shot": f"{s_zero}%",
                "One_Shot": r_one,
                "LLM_Judge_One_Shot": f"{s_one}%",
                "Few_Shot": r_few,
                "LLM_Judge_Few_Shot": f"{s_few}%",
                "Few_Shot_RAG": r_rag,
                "LLM_Judge_Few_Shot_RAG": f"{s_rag}%"
            })
            
        self._salvar_relatorio(resultados, scores)

    def _salvar_relatorio(self, resultados, scores):
        if not resultados:
            return
            
        media = lambda arr: round(sum(arr)/len(arr), 2) if arr else 0
        print("\n=== RESUMO DAS MÉTRICAS MÉDIAS (LLM-as-a-Judge) ===")
        print(f" -> Zero-Shot:      {media(scores['zero'])}%")
        print(f" -> One-Shot:       {media(scores['one'])}%")
        print(f" -> Few-Shot:       {media(scores['few'])}%")
        print(f" -> Few-Shot + RAG: {media(scores['few_rag'])}%")
        
        caminho_saida = DATASET_DIR / "relatorio_avaliacao.csv"
        with open(caminho_saida, mode="w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(resultados[0].keys()))
            writer.writeheader()
            writer.writerows(resultados)
        print(f"[+] Relatório salvo em: {caminho_saida}")
