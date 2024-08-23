import google.generativeai as genai
from openai import OpenAI
from meta_ai_api import MetaAI
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass
import boto3
from botocore.config import Config

config = Config(read_timeout=1000)

import os
_ = load_dotenv(find_dotenv())


@dataclass
class ModelConfig:
    temperature: float = 0.0
    top_p: float = 0.4
    max_tokens: int = 2048
    # max_output_tokens: int = 8192


def setup_google(modelId, system_message):
    genai.configure(api_key=os.environ["GEMINI_KEY"])
    # Create the model
    # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
    client = genai.GenerativeModel(
        model_name=modelId, system_instruction=system_message)
    return client


def setup_openAI():
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    return client


def setup_Bedrock():
    aws_access_key = os.environ['AWS_ACCESS_KEY']
    secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
    aws_region = os.environ['AWS_REGION']

    client = boto3.client(service_name='bedrock-runtime',
                          aws_access_key_id=aws_access_key,
                          aws_secret_access_key=secret_key,
                          region_name=aws_region,
                          config=config
                          )
    return client
