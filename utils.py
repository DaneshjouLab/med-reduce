import os
import json
import pickle
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

root = "/oak/stanford/groups/roxanad/"

def get_openai_client(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    return client

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4) 

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def get_root_dir():
    return root

def get_outputs_dir():
    return os.path.join(root, "rpark23/outputs/")

def get_clinical_dir():
    return os.path.join(root, "rpark23/clinical_data/tcga/")

def quarter_resolution(region_rgb):
    w, h = region_rgb.size
    return region_rgb.resize((w // 2, h // 2), resample=Image.BILINEAR)