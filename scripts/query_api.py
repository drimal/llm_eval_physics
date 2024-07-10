import os
from llm_eval_physics.data_loader import read_jsonl, prepare_question
from llm_eval_physics.config import ModelConfig
from llm_eval_physics.generator import generate
from llm_eval_physics.message_builder import MessageBuilder
from llm_eval_physics.utils import encode_image
import json
from dataclasses import asdict
from llm_eval_physics.utils import load_prompt_templates
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='llm_eval.log',
    encoding='utf-8',
    level=logging.DEBUG)


def main():
    prompts = load_prompt_templates()
    system_prompt = prompts.get("system_prompt")
    mcq_prompt = prompts.get("mcq_question_prompt")
    general_prompt = prompts.get("general_question_prompt")
    # model_provider = "aws"
    # modelId = "meta.llama3-70b-instruct-v1:0"
    # modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    # modelId = "mistral.mixtral-8x7b-instruct-v0:1"
    # model_provider = "openai"
    # modelId = "gpt-4o"
    model_provider = "google"
    modelId = "gemini-1.0-pro-latest"

    llm_params = asdict(ModelConfig())
    print(llm_params)

    indir = "../data/inputs/"
    outdir = "../data/outputs/"
    input_path = indir + "hseb12_modelqs.jsonl"
    images_dir = "../data/figures/"
    input_data = read_jsonl(input_path)
    questions = []
    output_fname = f"{outdir}/{modelId}_solutions_v2.md"
    f = open(output_fname, "w")
    f.write("# Answers Written by AI\n")
    for i, result in enumerate(input_data):
        print(f"Question: {i+1}\n")
        full_question, image_files = prepare_question(result)
        images = []
        base64_strings = []
        if image_files:
            for k, v in image_files.items():
                image = images_dir + v
                base64_strings.append(encode_image(image))

        f.write(f"### Question: {full_question}\n")
        # model_name = "gemini-1.0-pro-vision-latest"
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
        f.write(f"### Answer: {response}\n")
        # f.write(response["message"])

    js_text1 = """<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });</script>"""
    f.write(js_text1)
    f.close()
    os.system(f"mdpdf -o {output_fname.replace('.md', '.pdf')} {output_fname}")


if __name__ == "__main__":
    main()
