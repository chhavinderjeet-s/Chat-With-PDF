# RAG-based Chat with PDF (MLOps Enabled)

This project enables conversational interaction with PDF documents using a Retrieval-Augmented Generation (RAG) approach. Users can upload PDF files and ask questions about their content through a continuous chat interface. The system retrieves relevant document context and generates accurate, grounded responses while also evaluating answer quality and applying conditional deployment logic.

In addition to basic RAG functionality, this project includes evaluation metrics, rate limiting, and a simulated model registry to demonstrate core MLOps concepts such as model comparison and gated deployment.

## Features

- **PDF Upload**: Users can upload one or multiple PDF files containing the information they want to inquire about.

- **Text Extraction**: Extracts text content from uploaded PDF files for processing and analysis.

- **Text Chunking**: Splits the extracted text into smaller chunks for efficient processing and retrieval.

- **Vector Store Creation**: Utilizes FAISS to create a vector store from the text chunks, enabling fast and accurate retrieval of relevant information.

- **Continuous Chat Interface**: Supports multi-turn conversational interaction with uploaded documents.

- **Answer Evaluation**: Measures answer relevance and faithfulness using similarity-based metrics. 

- **Continuous Chat Interface**: Promotes a new model only if it outperforms the current production baseline.

- **Rate & Token Limiting**: Simulates free-tier usage constraints.

## Tech Stack
- **Python**: Programming language used for development.
- **Streamlit**: Web application framework for building interactive web applications.
- **PyPDF2**: Python library for reading PDF files.
- **Langchain**: Framework for developing RAG model using LLM.
- **FAISS**: Library for efficient similarity search and clustering of dense vectors.
- **OpenAI**: Used for LLM-based response generation.
- **spaCy**: Used for text embeddings and similarity calculations.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/sagnik-datta-02/ChatwithPDF.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Set up environment variables:

   - Add your OpenAI API key:`OPENAI_API_KEY`.
   
   - Make sure to have a `.env` file containing your environment variables, including the OpenAI API key.

4. Run the Streamlit app:

```bash
streamlit run app/main.py
```

## Usage

1. Upload one or more PDF files from the sidebar.
2. Click Submit & Process to process and embed the documents.
3. Ask questions in the chat input field.
4. View the generated answers along with evaluation metrics and deployment decision.


## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [PyPDF2](https://github.com/mstamy2/PyPDF2)
- [langchain](https://github.com/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [OpenAI](https://platform.openai.com/docs/overview)
- [spaCy](https://spacy.io)
- [python-dotenv](https://github.com/theskumar/python-dotenv)
