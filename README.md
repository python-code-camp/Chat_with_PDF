# Chat_with_PDF

Explanation : https://youtu.be/WmuSEfgzcJo

This project demonstrates how to utilize a Large Language Model (LLM) to process and understand PDF documents for question-answering tasks.

The implementation leverages several powerful libraries to achieve this:

Torch: For leveraging GPU acceleration in machine learning tasks.

Langchain: A suite of tools for loading documents, splitting text, generating embeddings, and more.

HuggingFace Embeddings: For creating semantic representations of text.

Chroma: For building a vector store that enables efficient similarity-based retrieval.

LlamaCpp: An LLM used for generating answers based on the processed text.

Conversational Retrieval Chain: For maintaining context and generating accurate responses during interaction.

------------------


Key Features

Device Adaptation: Automatically uses CUDA-enabled GPU if available, otherwise falls back to CPU.

Document Loading: Loads and processes PDF documents to prepare for analysis.

Text Splitting: Efficiently splits large texts into manageable chunks.

Embeddings and Vector Store: Utilizes HuggingFace embeddings and Chroma vector store for efficient information retrieval.

Conversational Interface: Engages users in a Q&A session using the LLM with context-aware responses.

------------------



How It Works

Load the Document: The PDF document is loaded and its content extracted.

Split the Text: The document is split into chunks to manage large texts effectively.

Initialize the Model: An LLM is configured to generate responses based on the processed text.

Create Vector Store: Text chunks are embedded and stored for quick retrieval.

Run the Interaction Loop: Users can ask questions, and the system provides accurate answers based on the document content.

------------------



Usage

Simply run the script, and interact with the system by asking questions about the document. The model will fetch and display relevant answers, demonstrating the power of LLMs in understanding and processing large texts.
