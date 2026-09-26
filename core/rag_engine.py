import os
import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from config import DOCS_DIR, VECTOR_STORE_DIR

MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class RAGEngine:
    def __init__(self):
        self.encoder = SentenceTransformer(MODELO_EMBEDDING)
        # Inicializa o ChromaDB com persistência no disco
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        self.collection = self.chroma_client.get_or_create_collection(name="agricultura_docs")

    def carregar_e_processar_pdfs(self) -> bool:
        """Lê, quebra semanticamente e armazena os PDFs no ChromaDB (apenas se a coleção estiver vazia)."""
        if self.collection.count() > 0:
            print(f"[+] Documentos já processados no ChromaDB ({self.collection.count()} chunks).")
            return True
            
        print("[+] Processando PDFs e criando embeddings... (Isso pode demorar um pouco)")
        loader = PyPDFDirectoryLoader(str(DOCS_DIR))
        documentos = loader.load()
        
        if not documentos:
            print(f"[-] Nenhum documento encontrado na pasta {DOCS_DIR}.")
            return False
            
        # Divisor semântico
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documentos)
        
        if not chunks:
            return False
            
        textos = [chunk.page_content for chunk in chunks]
        metadados = [{"fonte": f"{os.path.basename(chunk.metadata.get('source', ''))} (Pág. {chunk.metadata.get('page', 0)})"} for chunk in chunks]
        ids = [f"doc_{i}" for i in range(len(chunks))]
        
        # Gera embeddings usando SentenceTransformer
        embeddings = self.encoder.encode(textos, show_progress_bar=True).tolist()
        
        # Adiciona na coleção do ChromaDB em lotes para evitar limite máximo (ex: 5461)
        try:
            batch_size = self.chroma_client.get_max_batch_size()
        except AttributeError:
            batch_size = 5000
            
        for i in range(0, len(textos), batch_size):
            end_idx = i + batch_size
            self.collection.add(
                embeddings=embeddings[i:end_idx],
                documents=textos[i:end_idx],
                metadatas=metadados[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        print(f"[+] {len(chunks)} chunks adicionados ao ChromaDB.")
        return True

    def buscar_contexto(self, pergunta: str, top_k: int = 3) -> tuple[str, list[str]]:
        """Busca os textos mais relevantes usando ChromaDB."""
        if self.collection.count() == 0:
            return "", []
            
        embedding_pergunta = self.encoder.encode([pergunta]).tolist()
        
        resultados = self.collection.query(
            query_embeddings=embedding_pergunta,
            n_results=top_k
        )
        
        documentos = resultados.get("documents", [[]])[0]
        metadados = resultados.get("metadatas", [[]])[0]
        
        contexto_texto = ""
        fontes_usadas = []
        
        for doc, meta in zip(documentos, metadados):
            fonte = meta.get("fonte", "Desconhecida")
            contexto_texto += f"- [{fonte}]: {doc}\n\n"
            fontes_usadas.append(fonte)
            
        return contexto_texto.strip(), fontes_usadas
