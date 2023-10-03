import os
import logging
from model_wrapper import (load_model, HuggingFaceInstructEmbeddings, Chroma, PromptTemplate, ConversationBufferMemory, RetrievalQA)
from defines import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME
from dotenv import load_dotenv
from webex_bot.models.command import Command
from webex_bot.models.response import Response
import torch
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

log = logging.getLogger(__name__)

device_type = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"device type being used is: {device_type}")


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})


db = FAISS.load_local("PROCESSED_DATA", embeddings)
retriever = db.as_retriever()


## Default LLaMA-2 prompt style
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

instruction = """CONTEXT:/n/n {context}/n

Question: {question}"""
get_prompt(instruction, sys_prompt)


from langchain.prompts import PromptTemplate
prompt_template = get_prompt(instruction, sys_prompt)

llama_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": llama_prompt}


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

#prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
#memory = ConversationBufferMemory(input_key="question", memory_key="history")
llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':2}), #retriever,
    return_source_documents=True,
    #chain_type_kwargs={"prompt": prompt, "memory": memory},
    chain_type_kwargs=chain_type_kwargs,
)
print(qa)


class CXTMbot(Command):
    messages=[]
    messages.append({"role": "system", "content": "You are a polite assistant answering questions about Cisco collaboration and askCXTM."})

    def __init__(self):
        super().__init__()
        log.info("In gpt __init__ method")
    
    def execute(self, message, attachment_actions, activity):
        self.messages.append({"role": "user", "content": message})

        res = qa(message)
        answer = res["result"]

        self.messages.append({"role": "assistant", "content": answer})
        return answer