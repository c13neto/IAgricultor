# IAgricultor - Documentação Técnica e Arquitetural

Bem-vindo à documentação técnica do **IAgricultor**. Este documento foi criado para guiar futuros desenvolvedores (ou você mesmo no futuro) na manutenção e evolução da base de código. O projeto foi refatorado para sair de um script monolítico em direção a uma arquitetura de microsserviços internos, escalável e pronta para produção.

## 1. Arquitetura do Sistema

O projeto adota uma arquitetura modularizada, separando responsabilidades de interface, lógica de IA e avaliação de desempenho.

- `main.py`: Ponto de entrada do sistema. Responsável apenas por instanciar a interface de linha de comando (CLI) e rotear a requisição do usuário para o serviço apropriado.
- `config.py`: Centraliza todas as constantes e variáveis de ambiente globais.
- `core/`:
  - `llm_service.py`: Encapsula a lógica de carregamento, inferência e otimização do modelo de linguagem (LLM).
  - `memory.py`: Gerencia a janela de contexto deslizante, contando tokens ativamente para evitar o estouro de memória no prompt.
  - `rag_engine.py`: Gerencia a injeção de contexto externo. Lê PDFs, divide em *chunks* e armazena os vetores em um banco de dados persistente.
- `evaluation/`:
  - `evaluator.py`: Motor de testes (benchmark) assíncrono. Lê os dados de teste e dispara inferências *Zero-Shot*, *One-Shot*, *Few-Shot* e *RAG* para comparar abordagens.
  - `metrics.py`: Módulo responsável pela nota de desempenho. Utiliza *LLM-as-a-Judge* (onde a IA julga a precisão semântica de suas próprias respostas).

---

## 2. Glossário de Bibliotecas: Por que não usar apenas lógica básica de programação?

Em Inteligência Artificial Moderna, usar algoritmos básicos desenvolvidos do zero (`numpy`, `for loops`, `split(" ")`) para certas tarefas pode causar perda drástica de performance ou limitações inaceitáveis de hardware. Abaixo listamos as ferramentas avançadas utilizadas no projeto e os motivos técnicos para a sua adoção.

### `llama-cpp-python`
- **O que é?** Uma interface em Python (binding) que conecta nosso código a uma biblioteca nativa feita inteiramente em C/C++ puro.
- **Por que não usar lógica básica?** Processar bilhões de parâmetros matemáticos de uma Rede Neural de 3B parâmetros com Python nativo levaria horas por causa da lentidão de interpretação do Python (GIL). O `llama.cpp` desce a lógica para a camada de hardware, alocando diretamente as matrizes na memória RAM/VRAM e usando todos os núcleos físicos da CPU (hyperthreading) de forma muito rápida e otimizada, permitindo inferência local eficiente.

### `chromadb`
- **O que é?** Um Banco de Dados Vetorial (Vector Store) open-source e leve.
- **Por que não usar lógica básica?** Antigamente, nós calculávamos a "distância cosseno" (cosine similarity) entre vetores usando matrizes `numpy.dot()` diretamente na memória RAM toda vez que uma pergunta era feita. O problema é que, ao rodar milhares de páginas de PDF (ex: 7.846 chunks de texto), recalcular esses vetores consome GBs de memória e demora. O `ChromaDB` transforma essa matemática em um banco de dados **persistente em disco**. Ele guarda e indexa os vetores para buscas semânticas instantâneas sem recarregamento.

### `langchain-text-splitters` (RecursiveCharacterTextSplitter)
- **O que é?** Um fatiador avançado de documentos.
- **Por que não usar lógica básica?** O método ingênuo anterior cortava o texto em pedaços baseados estritamente em um número exato de caracteres (ex: `texto[i:i + 500]`). Isso cortava as palavras ao meio, separava frases cruciais em dois arquivos diferentes e arruinava o sentido semântico das dicas agrícolas. O fatiador do LangChain usa "recursividade inteligente", tentando quebrar por parágrafos primeiro, depois sentenças, preservando toda a coesão gramatical antes de transformar o texto em vetores.

### `sentence-transformers`
- **O que é?** Um encoder especializado que converte texto legível humano para representações vetoriais de alta dimensão.
- **Por que não usar lógica básica?** Python puro não "entende" semântica. O SentenceTransformers contém modelos pequenos (como o *all-MiniLM*) treinados exatamente para transformar uma frase inteira num vetor de dezenas de dimensões matemáticas. Isso permite que a busca do RAG encontre similaridades baseadas no "significado" das palavras e não na contagem exata da grafia.

### `tiktoken`
- **O que é?** Um contador rápido (escrito em Rust) de subpalavras algorítmicas (BPE - Byte Pair Encoding).
- **Por que não usar lógica básica?** Em sistemas anteriores, a memória do chatbot usava `len(texto.split())` para tentar adivinhar quando a conversa estava ficando grande demais. A Inteligência artificial não lê "palavras", lê "Tokens" (fragmentos que podem ser meia palavra ou símbolos matemáticos). O `tiktoken` conta cirurgicamente a quantidade de blocos que a IA vai processar, permitindo criar um "Sliding Window" na Memória (`memory.py`) sem o risco de o LLM quebrar o limite de 2.000 tokens e corromper o histórico.
