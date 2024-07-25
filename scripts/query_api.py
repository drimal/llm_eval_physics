import os
from llm_eval_physics import read_jsonl, prepare_question
from llm_eval_physics import ModelConfig
from llm_eval_physics import generate
from llm_eval_physics import MessageBuilder
from llm_eval_physics import encode_image
import argparse
import json
from dataclasses import asdict
from llm_eval_physics.utils import load_prompt_templates
import logging
import time

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='llm_eval.log',
    encoding='utf-8',
    level=logging.DEBUG)

def parse_arguements():
    available_models = ["meta.llama3-1-70b-instruct-v1:0","meta.llama3-70b-instruct-v1:0", "mistral.mixtral-8x7b-instruct-v0:1", "gemini-1.0-pro-latest", 'gpt-4o']
    parser = argparse.ArgumentParser()
    #parser.add_argument('-p', '--provider', required=True, type=str, help="Model Provider")
    parser.add_argument('-m', '--model', required=True, type=str, help=f"Model ID (model name) available models {available_models}")
    
    args = parser.parse_args()

    print("input args:\n", json.dumps(vars(args), indent=8, separators=(",", ":")))
    return args
    

def main(args):
    prompts = load_prompt_templates()
    system_prompt = prompts.get("system_prompt")
    mcq_prompt = prompts.get("mcq_question_prompt")
    general_prompt = prompts.get("general_question_prompt")
    modelId = args.model
    if modelId in ['gemini-1.0-pro-latest']:
        model_provider = "google"
    elif modelId in ['gpt-4o']:
        model_provider = "openai"
    else:
        model_provider = "aws"
    # modelId = "meta.llama3-70b-instruct-v1:0"
    # modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    # modelId = "mistral.mixtral-8x7b-instruct-v0:1"
    # modelId = "gpt-4o"
    #modelId = "gemini-1.0-pro-latest"

    llm_params = asdict(ModelConfig())
    print(llm_params)
    indir = f"../data/inputs/"
    outdir = "../data/outputs/"
    input_path = indir + "hseb12_modelqs.jsonl"
    images_dir = "../data/figures/"
    input_data = read_jsonl(input_path)
    output_fname = f"{outdir}/{modelId}_solutions_v2.md"
    f = open(output_fname, "w")
    f.write(f"# Answers Written by AI (Model: {modelId}) \n\n")
    f.write(" ")
    for i, result in enumerate(input_data):
        print(f"Question: {i+1}\n")
        full_question, image_files = prepare_question(result)
        images = []
        base64_strings = []
        if image_files:
            for k, v in image_files.items():
                image = images_dir + v
                base64_strings.append(encode_image(image))

        f.write(f"**Question:** {full_question} \n\n")
        f.write(" ")
        builder = MessageBuilder(system_prompt, mcq_prompt, general_prompt)
        messages = builder.create_messages(
            result["qtype"], full_question, images)
        messages = [{"role": msg.role, "content": msg.content}
                    for msg in messages]
        logger.info(messages)
        response = generate(
            provider=model_provider,
            modelId=modelId,
            model_config=llm_params,
            messages=messages,
            img_strings=base64_strings)
        # print([{"role": msg.role, "content": msg.content} for msg in messages])
        # response = client.messages.create(
        #    system = system_prompt,
        #    messages=,model="claude-3-opus-20240229", **llm_params)
        # f.write(response.choices[0].message.content)
        f.write(f"**Answer:** {response} \n\n")
        time.sleep(30)
        # f.write(response["message"])

    js_text1 = """<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });</script>"""
    f.write(js_text1)
    f.close()
    os.system(f"mdpdf -o {output_fname.replace('.md', '.pdf')} {output_fname}")

if __name__ == "__main__":
    args = parse_arguements()
    main(args)
