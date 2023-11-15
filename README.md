# Project

## Installation
### Download
```
git clone https://github.com/abdelrahmanemadismail/graduation-project
cd graduation-project
```
### Created and Use a virtual environment
```
python -m venv venv
.\venv\Scripts\activate
```
### Prepare env
Create `.env` file then add 
```
HUGGINGFACEHUB_API_TOKEN=<your_token>
```
you will find `<your_token>` at https://huggingface.co/settings/tokens

### install requirements
```
pip install pypdf2
pip install python-dotenv
pip install langchain
pip install faiss-cpu
pip install huggingface_hub
pip install InstructorEmbedding
pip install sentence_transformers
pip install spacy
python -m spacy download en_core_web_sm
```