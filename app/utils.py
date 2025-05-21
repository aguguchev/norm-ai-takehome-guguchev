from pydantic import BaseModel
import qdrant_client

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.schema import ( Document, Node )
from llama_index.core import ( Settings, VectorStoreIndex )
from dataclasses import dataclass

import os
import pymupdf4llm
import re

key = os.environ['OPENAI_API_KEY']

@dataclass
class Input:
    query: str
    file_path: str

@dataclass
class Citation:
    source: str
    text: str

class Output(BaseModel):
    query: str
    response: str
    citations: list[Citation]

class DocumentService:

    SECTION_NUMBER_BASE_PATTERN = r'\d+\.'
    ALL_HEADERS_REGEX = re.compile(r"\n\*\*(\w+\S*)\*\*")

    """
    Update this service to load the pdf and extract its contents.
    The example code below will help with the data structured required
    when using the QdrantService.load() method below. Note: for this
    exercise, ignore the subtle difference between llama-index's 
    Document and Node classes (i.e, treat them as interchangeable).

    # example code
    def create_documents() -> list[Document]:

        docs = [
            Document(
                metadata={"Section": "Law 1"},
                text="Theft is punishable by hanging",
            ),
            Document(
                metadata={"Section": "Law 2"},
                text="Tax evasion is punishable by banishment.",
            ),
        ]

        return docs

     """

    """
    generate_doc_list generates a list of docs given an input string
    in markdown format. String must satisfy the assumptions below:
        - Section headers must be bolded or titled (**<text>** or
            #... in markdown). Otherwise they will be processed as
            laws.
        - Laws must be written in nested format, e.g. '**1.** **Section 1**\n1.1. <1.1. text>...'
        - Each law or header must be on its own line
        - All text must be part of the nested structure, no non-legal
            text should be included. A title at the top is fine and
            will be treated as the "root section" in doc metadata

    :param self: self reference
    :param content: content string to parse into docs
    :param section_header: section header prefix for docs in content
    :param section_number: section number of law expressed as a list, e.g. 1.4 = [1, 4]
    :return: generated list of documents

    """

    # TODO: Clean this up
    def generate_doc_list(self, content: str, section_header="", section_number=[]):
        depth = len(section_number)
        section_pattern_str = f"\n\\**{self.SECTION_NUMBER_BASE_PATTERN * (depth + 1)}\\**\\D"
        section_pattern = re.compile(section_pattern_str)

        split_content = re.split(section_pattern, content)

        # Base case - content is of form '<text>' with all preceding section numbers and headers
        # passed via params
        if len(split_content) == 1:
            return [Document(
                metadata={"SectionPath": section_header,
                          "SectionNumber": ".".join(section_number) + "."},
                text=" ".join(split_content[0].split()).strip()
            )]

        # First element of split_content is always the section header for any nested laws
        # Remove formatting artifacts before passing as section_header arg
        sub_section_header = " ".join(split_content[0].split('\n'))
        if sub_section_header.startswith('**'):
            sub_section_header = sub_section_header[2:-2]
        elif sub_section_header.startswith('# '):
            sub_section_header = sub_section_header[2:]

        final_doc_list = []
        # Sometimes a subsection header may be a law in its own right (e.g. 3.1.)
        # Needs to be extracted as its own doc
        if not (split_content[0].startswith('#') or split_content[0].startswith('**')):
            final_doc_list.extend(self.generate_doc_list(content=split_content[0],
                                            section_header=section_header,
                                            section_number=section_number))

        for i in range(1, len(split_content)):
            token = split_content[i]
            section_header_stub = section_header + "/" if len(section_header) > 0 else ""
            final_doc_list.extend(self.generate_doc_list(content=token,
                                            section_header=section_header_stub + sub_section_header,
                                            section_number=section_number + [str(i)]))

        return final_doc_list


    def read_docs_from_path(self, doc_path: str) -> list[Document]:
        if not doc_path.endswith(".pdf"):
            raise ValueError("Doc path does not point to a valid pdf file. Only PDF files are supported.")

        md = pymupdf4llm.to_markdown(doc_path).split("\n")

        # Preprocess markdown file
        # Remove empty lines
        md = "\n".join(list(filter(lambda x: x != "", md)))

        # Find last law section and trim all footer content past it
        top_level_section_regex = re.compile("\*\*" + self.SECTION_NUMBER_BASE_PATTERN + "\*\*")
        last_section_match = list(re.finditer(top_level_section_regex, md))[-1].end()
        footer_start_index = self.ALL_HEADERS_REGEX.search(md, last_section_match).start()
        md = md[:footer_start_index]

        doc_list = self.generate_doc_list(md)

        return doc_list

    def create_documents(self, doc_root: str) -> list[Document]:
        # Read docs from all relevant paths
        doc_list = []
        for root, dirs, file_names in os.walk(doc_root):
            print(f"Files in {root}: {file_names}")
            for name in file_names:
                if name.endswith(".pdf"):
                    doc_path = os.path.join(root, name)
                    print(f"Reading docs from {doc_path}")
                    doc_list.extend(self.read_docs_from_path(doc_path))

        return doc_list

class QdrantService:

    GET_SOURCE_NUM_REGEX = re.compile(r"Source (\d+):\n")

    def __init__(self, k: int = 2):
        self.index = None
        self.k = k
    
    def connect(self) -> None:
        client = qdrant_client.QdrantClient(location=":memory:")
        vstore = QdrantVectorStore(client=client, collection_name='temp')

        Settings.embed_model = OpenAIEmbedding(api_key=key)
        Settings.llm = OpenAI(api_key=key, model='gpt-4o-mini')

        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vstore
            )

    def load(self, docs: list[Document]):
        self.index.insert_nodes(docs)

        self.query_engine = CitationQueryEngine.from_args(
            index=self.index,
            similarity_top_k=self.k
        )

    def create_citation_from_node(self, node: Node):
        source_match = re.search(self.GET_SOURCE_NUM_REGEX, node.get_text())

        return Citation(
            source=source_match.group(1),
            text=node.get_text()[source_match.end(0):].strip()
        )
    
    def query(self, query_str: str) -> Output:

        """
        This method needs to initialize the query engine, run the query, and return
        the result as a pydantic Output class. This is what will be returned as
        JSON via the FastAPI endpount. Fee free to do this however you'd like, but
        a its worth noting that the llama-index package has a CitationQueryEngine...

        Also, be sure to make use of self.k (the number of vectors to return based
        on semantic similarity).

        # Example output object
        citations = [
            Citation(source="Law 1", text="Theft is punishable by hanging"),
            Citation(source="Law 2", text="Tax evasion is punishable by banishment."),
        ]

        output = Output(
            query=query_str, 
            response=response_text, 
            citations=citations
            )
        
        return output

        """
        response = self.query_engine.query(query_str)
        citations = [self.create_citation_from_node(node.node) for node in response.source_nodes ]
        return Output(
            query=query_str,
            response=str(response),
            citations=citations
        )

       

if __name__ == "__main__":
    # Example workflow
    doc_serivce = DocumentService() # implemented
    docs = doc_serivce.create_documents() # NOT implemented

    index = QdrantService() # implemented
    index.connect() # implemented
    index.load() # implemented

    index.query("what happens if I steal?") # NOT implemented





