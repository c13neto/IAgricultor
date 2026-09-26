# rag_engine.py
import os
import glob
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class RAGEngine:
    def __init__(self, pasta_documentos="documentos"):
        self.pasta_documentos = pasta_documentos
        self.encoder = SentenceTransformer(MODELO_EMBEDDING)
        self.chunks = []
        self.embeddings = None

    def carregar_e_processar_pdfs(self, tamanho_chunk=500, sobreposicao=50):
        """Lê os arquivos PDF da pasta informada, divide em blocos (chunks) e gera os embeddings."""
        arquivos_pdf = glob.glob(os.path.join(self.pasta_documentos, "*.pdf"))
        if not arquivos_pdf:
            print(f"[-] Nenhum arquivo PDF encontrado na pasta '{self.pasta_documentos}'.")
            return False

        textos_extraidos = []
        for caminho_pdf in arquivos_pdf:
            nome_arquivo = os.path.basename(caminho_pdf)
            try:
                reader = PdfReader(caminho_pdf)
                for num_pagina, pagina in enumerate(reader.pages, start=1):
                    texto = pagina.extract_text()
                    if texto:
                        textos_extraidos.append({
                            "fonte": f"{nome_arquivo} (Pág. {num_pagina})",
                            "texto": texto
                        })
            except Exception as e:
                print(f"[-] Erro ao ler o arquivo {nome_arquivo}: {e}")
                continue

        self.chunks = []
        for doc in textos_extraidos:
            texto = doc["texto"]
            fonte = doc["fonte"]
            for i in range(0, len(texto), tamanho_chunk - sobreposicao):
                trecho = texto[i:i + tamanho_chunk].strip()
                if len(trecho) > 50:
                    self.chunks.append({
                        "fonte": fonte,
                        "texto": trecho
                    })

        if not self.chunks:
            print("[-] Nenhum texto válido foi extraído dos PDFs.")
            return False

        textos_chunks = [c["texto"] for c in self.chunks]
        self.embeddings = self.encoder.encode(textos_chunks, show_progress_bar=False)
        return True

    def buscar_contexto(self, pergunta, top_k=3):
        """Busca os top_k trechos mais relevantes do PDF para responder à pergunta."""
        if self.embeddings is None or len(self.chunks) == 0:
            return "", []

        embedding_pergunta = self.encoder.encode([pergunta])[0]
        
        similitudes = np.dot(self.embeddings, embedding_pergunta) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(embedding_pergunta)
        )
        
        indices_top_k = np.argsort(similitudes)[::-1][:top_k]
        
        contexto_texto = ""
        fontes_usadas = []

        for idx in indices_top_k:
            chunk = self.chunks[idx]
            contexto_texto += f"- [{chunk['fonte']}]: {chunk['texto']}\n\n"
            fontes_usadas.append(chunk['fonte'])

        return contexto_texto.strip(), fontes_usadas