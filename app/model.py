from pydantic import BaseModel

class QueryRequestV1(BaseModel):
    query_str: str