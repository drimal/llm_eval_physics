import json
from llm_eval_physics.config import setup_Bedrock, setup_google, setup_openAI
import tiktoken
from typing import List, Any
import time

def get_num_tokens(prompt: str, encoding_name: str = "r50k_base")->int:
    """returns approximate number of tokens in the prompt 

    Parameters
    ----------
    prompt : str
        
    encoding_name : str, optional
        name of tokenizer, by default "r50k_base"

    Returns
    -------
    int
        number of tokens
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(prompt))
    return num_tokens


def invoke_bedrock(
        client: Any,
        modelId: str,
        model_config: dict,
        prompt: str,
        img_strings: List[str])->str:
    """function to invoke aws bedrok

    Parameters
    ----------
    client : Any
        boto3 client
    modelId : str
        Id of the model to be used for e.g. "meta.llama3-70b-instruct-v1:0"
    model_config : dict
        dictionary of LLM parameters
    prompt : str
        input prompt 
    img_strings : List[str]
        List of base64 strings of the images

    Returns
    -------
    str
        model response
    """
    n_tokens = get_num_tokens(prompt)
    if "llama3" in modelId:
        max_token_param = "max_gen_len"
        model_outstring = "generation"
        if n_tokens > 8000:
            prompt = prompt[:8000] + \
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    elif "mistral" in modelId:
        max_token_param = "max_tokens"
        model_outstring = "outputs"
        if n_tokens > 8000:
            prompt = prompt[:4000] + "[/INST]"
    else:
        max_token_param = "max_tokens_to_sample"
        model_outstring = "completion"

    body = json.dumps({"prompt": prompt,
                       f"{max_token_param}": model_config["max_tokens"],
                       "temperature": model_config["temperature"],
                       "top_p": model_config["top_p"]})
    if modelId in ["anthropic.claude-3-sonnet-20240229-v1:0"]:
        contents = [
            {
                "type": "text",
                "text": prompt
            },
        ]
        if img_strings:
            for img_string in img_strings:
                contents.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_string
                        }
                    },
                )
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": model_config["max_tokens"],
            "temperature": model_config["temperature"],
            "messages": [
                {
                    "role": "user",
                    "content": contents
                }
            ],
        })
    retry_counter = 1
    while retry_counter <= 5:
        try:
            model_response = client.invoke_model(
                body=body,
                modelId=modelId,
                accept='application/json',
                contentType='application/json')
            response_body = json.loads(
                model_response.get('body').read().decode('utf-8'))
            if "anthropic.claude-3" in modelId:
                resp = response_body['content'][0]['text']
            elif "mistral" in modelId:
                resp = response_body[model_outstring][0]['text']
            else:
                resp = response_body.get(model_outstring).strip()
            return resp
        except BaseException as e:
            print(e)
            time.sleep(60)
            retry_counter += 1
    return ""
        


def invoke_openai(client, modelId: str, model_config: dict, messages):
    response = client.chat.completions.create(
        model=modelId,
        messages=messages,
        temperature=model_config["temperature"],
        top_p=model_config["top_p"],
        max_tokens=model_config["max_tokens"]
    )
    return response.choices[0].message.content


def invoke_google(client, modelId: str, model_config: dict, messages):
    prompt = format_gemini_messages(messages)
    gen_config = {
        "temperature": model_config['temperature'],
        "max_output_tokens": model_config['max_tokens'],
        "top_p": model_config['top_p']}
    response = client.generate_content([prompt], generation_config=gen_config)
    return response.text


def generate(
        provider,
        modelId: str,
        model_config: dict,
        messages,
        img_strings):
    if provider == "google":
        system_instructions = "You are a helpful assistant."
        if messages[0]["role"] == "system":
            system_instructions = messages[0]["content"]
        client = setup_google(modelId, system_message=system_instructions)
        return invoke_google(client, modelId, model_config, messages)
    elif provider == "openai":
        client = setup_openAI()
        return invoke_openai(client, modelId, model_config, messages)
    elif provider == "aws":
        client = setup_Bedrock()
        if "llama3" in modelId:
            messages = format_llama_messages(messages)
        elif "anthropic" in modelId:
            messages = format_anthropic(messages)
        elif "mistral" in modelId:
            messages = format_mistral(messages)
        return invoke_bedrock(
            client,
            modelId,
            model_config,
            messages,
            img_strings)
    else:
        raise Exception(NotImplementedError)


def format_llama_messages(messages):
    system_message = "".join([msg["content"]
                             for msg in messages if msg["role"] == "system"])
    user_message = "".join([msg["content"]
                           for msg in messages if msg["role"] == "user"])

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    return prompt


def format_anthropic(messages):
    system_message = "".join([msg["content"]
                             for msg in messages if msg["role"] == "system"])
    user_message = "".join([msg["content"]
                           for msg in messages if msg["role"] == "user"])
    prompt = f"""Human: {system_message} \n\n {user_message}\n Assistant: \n"""
    return prompt


def format_mistral(messages):
    system_message = "".join([msg["content"]
                             for msg in messages if msg["role"] == "system"])
    user_message = "".join([msg["content"]
                           for msg in messages if msg["role"] == "user"])
    return f"""<s> [INST]{system_message} {user_message} [/INST]"""


def format_gemini_messages(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "user":
            prompt += message["content"]
    return prompt
