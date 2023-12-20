import torch
import transformers


from transformers import LlamaForCausalLM, LlamaTokenizer
import os

def get_llm(model_name = 'meta-llama/Llama-2-7b-chat-hf'):
    
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16 # bfloat16
    )

    # begin initializing HF items, need auth token for these
    # hf_auth = ''
    model_config = transformers.AutoConfig.from_pretrained(
        model_name,
        # use_auth_token=hf_auth
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        # use_auth_token=hf_auth
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )
    
    return generate_text