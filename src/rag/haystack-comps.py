import os

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

HF_HOME = "G:\huggingface"
os.environ["HF_DATASETS_CACHE"] = HF_HOME
os.environ["HF_HOME"] = HF_HOME


document_store = ChromaDocumentStore(persist_path=".")

text_file_converter = TextFileToDocument()
cleaner = DocumentCleaner(remove_extra_whitespaces=True)
splitter = DocumentSplitter(split_length=8, split_by="sentence", split_overlap=2)
embedder = SentenceTransformersDocumentEmbedder(
    model="mixedbread-ai/mxbai-embed-large-v1", progress_bar=True
)
embedder.warm_up()

writer = DocumentWriter(document_store)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", text_file_converter)
indexing_pipeline.add_component("cleaner", cleaner)
indexing_pipeline.add_component("splitter", splitter)
indexing_pipeline.add_component("embedder", embedder)
indexing_pipeline.add_component("writer", writer)

indexing_pipeline.connect("converter.documents", "cleaner.documents")
indexing_pipeline.connect("cleaner.documents", "splitter.documents")
indexing_pipeline.connect("splitter.documents", "embedder.documents")
indexing_pipeline.connect("embedder.documents", "writer.documents")
indexing_pipeline.run(data={"sources": ["davinci.txt"]})

text_embedder = SentenceTransformersTextEmbedder(
    model="mixedbread-ai/mxbai-embed-large-v1", progress_bar=True
)
retriever = ChromaEmbeddingRetriever(document_store)
template = """Given these documents, answer the question.
              Documents:
              {% for doc in documents %}
                  {{ doc.content }}
              {% endfor %}
              Question: {{query}}
              Answer:"""
prompt_builder = PromptBuilder(template=template)
generator = HuggingFaceLocalChatGenerator()
generator.warm_up()


rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", generator)

rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

query = "How old was he when he died?"
result = rag_pipeline.run(
    data={"prompt_builder": {"query": query}, "text_embedder": {"text": query}}
)
print(result["llm"]["replies"][0])
