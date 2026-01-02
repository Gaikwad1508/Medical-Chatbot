# Medical-Chatbot-with-LLMs-LangChain-Pinecone-Groq-HuggingFace

# 🚀 [Click Here to Try the Live App](https://huggingface.co/spaces/Abhishek1508/medical-chatbot)

---

## How to run locally?
#### Steps:

Clone the repository

```bash
git clone [https://huggingface.co/spaces/Abhishek1508/medical-chatbot.git](https://huggingface.co/spaces/Abhishek1508/medical-chatbot)
```


## Step 1: Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```

## Step 2: Install the requirements

```bash
pip install -r requirements.txt
```

## Step 3: Create a .env file in the root directory and add your credentials

```bash
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## Step 4: Store embeddings to Pinecone

```bash
python store_index.py
```

## Step 5: Run the application

```bash
python app.py
```

Now, 
```bash
open up localhost:8080
```

### Techstack Used:
* Python
* LangChain
* Flask
* Groq (Llama 3.3)
* Pinecone
* Hugging Face Spaces

## Deployment-on-Hugging-Face-Spaces-with-Docker

### 1. Login to Hugging Face
### 2. Create New Space for deployment

```bash
#with specific configuration

1. Space Name : medical-chatbot

2. SDK: Docker (This is required for Flask apps)


#Description: About the deployment

1. Create a Dockerfile in your root directory

2. Push your code + Dockerfile to Hugging Face Space

3. Hugging Face builds the image automatically

4. The container runs your app.py on port 7860

```

### 3. Configure Secrets (Environment Variables)

```bash
- Go to Settings > Variables and secrets
- Add the following secrets:
  - PINECONE_API_KEY
  - GROQ_API_KEY

```

### 4. Repository Structure (Files to upload)

```bash
- src/ (folder)
- static/ (folder)
- templates/ (folder)
- data/ (folder - optional)
- app.py
- requirements.txt
- Dockerfile
- store_index.py (optional)
```

### 5. Dockerfile Configuration:

```bash
# Use Python 3.9 image
FROM python:3.9

# Set the working directory
WORKDIR /code

# Copy requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application
COPY . /code

# Set permissions and path
RUN mkdir -p /code/cache && chmod -R 777 /code/cache
ENV TRANSFORMERS_CACHE=/code/cache
ENV PYTHONPATH=/code

# Command to run the application
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
```

### 6. Build and Run:
```bash
Commit changes to main > The space will automatically build > Wait for "Running" status
```