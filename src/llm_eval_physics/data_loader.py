import json
from typing import Dict, Any

def read_jsonl(path: str) -> Dict[Any]:
    with open(path, encoding='utf-8') as jnf:
        results = []
        for line in jnf:
            if line is None:
                continue
            try:
                results.append(json.loads(line) if line != "null" else line)
            except Exception as e:
                print(e)
                print(path)
                print(line)
                raise e
        return results

def prepare_question(question_dict):
    qtype = question_dict.get("qtype")
    num = question_dict.get("num")
    full_question = f"Q{num}. "
    question = question_dict.get("question")
    options = question_dict.get("options")
    images = question_dict.get("images")

    if images:
        option_fig = images.get("option_fig")
        qfig = images.get("qfig")
        if qfig:
            question = question.format(placeholder=qfig)
        if option_fig and options == "img":
            options = option_fig
    full_question += question
    if qtype == "mcq":
        full_question += f" Options:  {str(options)}"
    return full_question, images
