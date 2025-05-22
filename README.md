This repository contains a FastAPI service that will answer questions about the laws and customs of Westeros when provided natural language queries from a user.

## How to launch the service
First, clone the repository locally with
```
git clone https://github.com/aguguchev/norm-ai-takehome-guguchev.git
```
### Method 1: Run the service locally
The most basic way to launch the service is to run the server locally. This can be done with
```
python3 <your-dest>/norm-ai-takehome-guguchev/run.py
```

### Method 2: Run the service in a docker container
You can also run the service in a docker container by executing the following commands:
```
# For Windows users
docker build -t norm-demo --build-arg OPENAI_API_KEY=$env:OPENAI_API_KEY .
# For macOS/Linux users
docker build -t norm-demo --build-arg OPENAI_API_KEY=$OPENAI_API_KEY .

docker container run -p 8080:80 norm-demo:latest
```

#### Sanity check
If the server is running correctly, the following command should yield a 200 status code:
```
curl -X GET http://localhost:8080/ping
```

## How to interact with the service
You can access a Swagger UI for the service at the url [localhost:8080/docs](localhost:8080/docs)

Similar to the sanity check, you can also make cURL requests from your machine against the query service. The pattern is below:
```
curl http://localhost:8080/executeQuery -X POST -H "Content-Type:application/json" -d "{\"query_str\":\"<YOUR-QUERY-HERE>\"}"
```

## Customizing the source material
The service will attempt to parse any document under the `/docs/` directory as long as it has the `.pdf` extension. To ensure best results, please follow the formatting of the existing `laws.pdf` file. That is:
- Section headers must be bolded or titled (**<text>** or #... in markdown). Otherwise they will be processed as laws.
- Laws must be written in nested format, e.g. '**1.** **Section 1**\n1.1. <1.1. text>...'
  - All section headers and laws must be numbered according to the nesting convention
- Each law or header must be on its own line
- All text must be part of the nested structure, no non-legal text should be included. A title at the top is fine and will be treated as the "root section" in doc metadata
  - Footer content is acceptable but should start with a title or header that is not numbered.

Currently the doc parsing logic is written with some basic regex that assumes the patterns of the provided `laws.pdf` doc are universal to the input space. In future iterations, this should be replaced with a more robust parser that can account for formatting variations. 
