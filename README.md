# Multi-Model PDF Intelligence Assistant

A powerful RAG (Retrieval-Augmented Generation) application that allows you to chat with your PDF documents using multiple AI models. This application uses Streamlit for the interface and LangChain for document processing and AI interactions.

## Features

- 📄 PDF document processing and analysis
- 🤖 Multiple AI model support (GPT-4, GPT-3.5-Turbo, GPT-4-Turbo)
- 🔍 Advanced document chunking and retrieval
- 💬 Interactive chat interface
- 🎯 Semantic search capabilities

## Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

5. Run the application:
```bash
streamlit run rag_pdfloader.py
```

## Usage

1. Upload a PDF document using the file uploader
2. Select your preferred AI model from the dropdown menu
3. Start asking questions about your document
4. View the conversation history below

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in requirements.txt

## License

MIT License 