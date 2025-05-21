import uvicorn

from fastapi import FastAPI, Query
from http import HTTPStatus
from app.utils import DocumentService
from app.utils import QdrantService
from app.utils import Output
from app.model import QueryRequestV1

app = FastAPI()
index = QdrantService()

"""
Please create an endpoint that accepts a query string, e.g., "what happens if I steal 
from the Sept?" and returns a JSON response serialized from the Pydantic Output class.
"""

@app.get("/ping")
def ping():
    return HTTPStatus.OK

@app.post("/executeQuery", response_model=Output)
def execute_query(query_req: QueryRequestV1):
    return index.query(query_req.query_str)

def main():
    print("Creating documents")
    doc_serv = DocumentService()
    docs = doc_serv.create_documents("docs")

    index.connect() # implemented
    index.load(docs) # implemented

    uvicorn.run(app=app, host='0.0.0.0', port=80)
