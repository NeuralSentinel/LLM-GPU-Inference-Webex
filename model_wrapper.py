import logging

import click
import torch
#from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from torch import cuda, bfloat16
import transformers


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

from defines import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME


def load_model(device_type, model_id, model_basename=None):
    hf_auth = 'YOUR KEY'
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    print("*************************************************************Inside Load Model")
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for GGML quantized models")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            max_ctx_size = 2048
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)

        else:
            print("************************************Inside IF then else")
            # The code supports all huggingface models that ends with GPTQ and have some variation
            # of .no-act.order or .safetensors in their HF repo.
            logging.info("Using AutoGPTQForCausalLM for quantized models")

            if ".safetensors" in model_basename:
                # Remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")


            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            logging.info("Tokenizer loaded")

            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                #device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
    elif (
   
        device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        print("************************************Inside elif")
        logging.info("Using AutoModelForCausalLM for full models")
        bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bfloat16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = "../models", use_auth_token=hf_auth)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            #low_cpu_mem_usage=True,
            trust_remote_code=True,
            #load_in_4bit = True,
            use_auth_token=hf_auth, 
            cache_dir = "../models",
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        #model.tie_weights()
        model.to_bettertransformer()
    else:
        print("************************************Inside else")
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id, cache_dir = "../models", use_auth_token=hf_auth)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        #max_length=4096,
        max_new_tokens = 1024,
        #temperature=0.0,
        #top_p=0.95,
        top_k = 10,
        #repetition_penalty=1.15,
        generation_config=generation_config,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0})
    logging.info("Local LLM Loaded")

    return local_llm