from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llmmodel = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task='text-generation'
)

model = ChatHuggingFace(llm=llmmodel)
result = model.invoke("Can you tell me capital of france?")
print(result.content)