from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import GrobidParser
loader = GenericLoader.from_filesystem(
    "../Papers/",
    glob="*",
    suffixes=[".pdf"],
    parser=GrobidParser(segment_sentences=False),
)
docs = loader.load()
print(docs[0].page_content)