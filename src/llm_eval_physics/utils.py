import yaml
import os
import base64


def load_prompt_templates(version="1.0.0"):
    thisdir = os.path.dirname(os.path.abspath(__file__))
    template_yaml_file = open(os.path.join(thisdir, "prompt_template.yaml"))
    templates = yaml.safe_load(template_yaml_file)
    prompts = templates.get("prompts").get("version").get(version)
    return prompts


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
