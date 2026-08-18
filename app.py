from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline


# 1. Load Embedding Model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Load FAISS Vector Store

vector_store = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)


# 3. Load Hugging Face LLM

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=300,
    temperature=0.2
)


# 4. Get User Question

query = input("\nAsk a question: ")


# 5. Retrieve Relevant Chunks

results = vector_store.similarity_search(
    query,
    k=3
)


# 6. Create Context

context = "\n\n".join(
    result.page_content
    for result in results
)


# 7. Create Prompt

prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the context provided below.

If the answer is not present in the context,
say "I don't know."

Context:
{context}

Question:
{query}

Answer:
"""


# 8. Generate Answer

output = generator(
    prompt,
    return_full_text=False
)


# 9. Display Answer

answer = output[0]["generated_text"]

print("\nAnswer:")
print(answer)


# 10. Display Sources

print("\nSources:")

for result in results:

    source = result.metadata.get("source")
    page = result.metadata.get("page")

    print(f"- {source} | Page {page + 1}")