# Basic RAG from Scratch

A simple implementation of a **Retrieval-Augmented Generation (RAG)** pipeline using Python, Sentence Transformers, FAISS, and Groq — without using LangChain.

This project demonstrates how to build a basic RAG system that retrieves relevant information from a research paper and uses a Large Language Model (LLM) to generate context-aware answers.

## Project Overview

The project uses the research paper **"VQA: Visual Question Answering"** as its knowledge source.

Users can ask questions about the document, and the system retrieves relevant text chunks before passing them to an LLM to generate answers.

### RAG Pipeline

1. **Document Ingestion:** Extract text from a PDF using PyMuPDF.
2. **Text Chunking:** Combine document pages and split the text into overlapping token-based chunks while preserving page metadata.
3. **Embedding Generation:** Convert text chunks into vector embeddings using Sentence Transformers.
4. **Vector Storage:** Store embeddings in a FAISS vector index.
5. **Retrieval:** Retrieve the most relevant chunks based on semantic similarity to the user's query.
6. **Augmentation:** Combine retrieved chunks with the original query to construct a prompt.
7. **Generation:** Use an LLM through Groq to generate an answer based on the retrieved context.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyMuPDF | PDF text extraction |
| tiktoken | Tokenization and chunking |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Embedding model |
| FAISS | Vector indexing and similarity search |
| Groq | LLM inference API |
| GPT-OSS-20B | Answer generation |
| NumPy | Numerical operations |

## Project Structure

```text
Basic-RAG-from-Scratch/
│
├── Basic_RAG_from_Scratch.ipynb
├── VQA_Visual_Question_Answering.pdf
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Basic-RAG-from-Scratch
```

### 2. Install dependencies

```bash
pip install pymupdf tiktoken sentence-transformers faiss-cpu groq numpy
```

### 3. Configure the Groq API key

Create an API key through the [Groq Console](https://console.groq.com/).

Set the API key as an environment variable rather than hardcoding it in the notebook.

For Google Colab, you can store it in Colab Secrets and access it with:

```python
from google.colab import userdata

GROQ_API_KEY = userdata.get("GROQ_API_KEY")
```

Initialize the Groq client:

```python
from groq import Groq

llm = Groq(api_key=GROQ_API_KEY)
```

### 4. Run the notebook

Open `Basic_RAG_from_Scratch.ipynb` in Google Colab or Jupyter Notebook.

Make sure the PDF research paper is available at the path specified in the notebook.

Run the notebook cells sequentially to build and test the RAG pipeline.

## Implementation Details

### Document Chunking

- Chunk size: 200 tokens
- Overlap: 30 tokens
- Tokenizer: `cl100k_base`
- Page metadata is preserved for source identification.
- Chunks are verified against the embedding model's maximum sequence length.

### Embedding Model

The project uses `all-MiniLM-L6-v2` to convert text chunks into 384-dimensional embeddings.

### Vector Database

FAISS `IndexFlatL2` is used for exact L2 distance-based similarity search.

### Answer Generation

The retrieved document context is passed to Groq's `openai/gpt-oss-20b` model, which generates answers using the supplied context.

The prompt instructs the model to:
- Answer using only the retrieved context.
- Include relevant page references.
- Avoid inventing information.
- Indicate when the answer is unavailable in the retrieved context.

## Retrieval Evaluation

The retriever is evaluated using three metrics:

- **Precision@K:** Measures the proportion of retrieved chunks that are relevant.
- **Recall@K:** Measures the proportion of relevant chunks successfully retrieved.
- **Mean Reciprocal Rank (MRR):** Measures how highly the first relevant chunk is ranked.

The evaluation uses three manually annotated test queries and compares retrieval performance at K = 1, 3, and 5.

### Results

| Metric | K = 1 | K = 3 | K = 5 |
|---|---:|---:|---:|
| Precision | 1.000 | 0.556 | 0.400 |
| Recall | 0.361 | 0.556 | 0.722 |
| MRR | 1.000 | 1.000 | 1.000 |

**Observations:**

- Increasing K improves recall by retrieving more relevant chunks.
- Precision decreases as more chunks are retrieved.
- MRR remains 1.0 across all three K values because the first retrieved chunk is relevant for each test query.

*Note: These results are based on a small evaluation set of three queries with manually assigned relevant chunk IDs. They demonstrate the implementation and are not a general benchmark of retrieval performance.*

## Example Queries

The system was tested with questions such as:

- What is Visual Question Answering?
- What datasets are used for VQA?
- What are the challenges of Visual Question Answering?

The generated answers include relevant information from the research paper and associated page references.

## Learning Outcomes

Through this project, I explored and implemented the core components of a RAG pipeline without relying on a framework such as LangChain.

Key concepts learned:

- PDF document ingestion and preprocessing.
- Token-based text chunking with overlap.
- Preserving document metadata across chunks.
- Generating semantic embeddings.
- Vector indexing and similarity search with FAISS.
- Query-based retrieval.
- Context augmentation and prompt construction.
- LLM-based answer generation.
- Evaluating retrieval performance using Precision@K, Recall@K, and MRR.

## Future Improvements

- Support multiple PDF documents.
- Improve source citation verification.
- Experiment with different embedding models.
- Expand the retrieval evaluation dataset.

---

**Project Type:** Learning / Practical Implementation  
**Focus:** Understanding and implementing a basic RAG pipeline from scratch.