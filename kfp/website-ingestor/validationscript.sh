#add below line in ~/.bashrc
#alias python=/opt/homebrew/bin/python3

python3 -m venv venv 
source ./venv/bin/activate



pip install requests beautifulsoup4 weaviate-client langchain transformers langchain-community sentence-transformers einops html2text openai


#oc port-forward svc/weaviate-vector-db 55556:8080 &

export WEAVIATE_API_KEY=""
export WEAVIATE_HOST="http://localhost"
export WEAVIATE_PORT="55556"

export API_URL="https://granite-8b-code-instruct-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443"
export API_KEY=""



python3 ./kfp/website-ingestor/ingestion-pipeline-website-local.py
