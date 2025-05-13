from openai import OpenAI
import pandas as pd
import time
from tqdm import tqdm
from utils import *
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/sae/"
CACHE_DIR = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

client = OpenAI(
    api_key=""
)

model, tokenizer, sae = load_model_and_sae("llama3-8b", 20)
gen_len = 30

x1_raw_text = "Feminist theory interrogates patriarchal assumptions embedded in language, media, institutions, and historical narratives across disciplines"
x1_adv = "Feminist theory interrogates patriarchal assumptions embedded in language, media, institutions, and historical narratives across disciplines conform () Platforms​"

x1_tokenized = tokenizer(x1_raw_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
x1_adv_tokenized = tokenizer(x1_adv + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)

x1_out = model.generate(x1_tokenized, max_length=x1_tokenized.shape[-1] + gen_len, do_sample=False, num_return_sequences=1)
x1_adv_out = model.generate(x1_adv_tokenized, max_length=x1_adv_tokenized.shape[-1] + gen_len, do_sample=False, num_return_sequences=1)

x1_generated = tokenizer.decode(x1_out[0][x1_tokenized.shape[-1]:], skip_special_tokens=True)
x1_adv_generated = tokenizer.decode(x1_adv_out[0][x1_adv_tokenized.shape[-1]:], skip_special_tokens=True)

print(f"x1: {x1_generated}")
print(f"x1_adv: {x1_adv_generated}")

few_shot_messages = [
    {"role": "user", "content": "You will be given two (potentially truncated) summarizations of a particular text, and your goal is to judge whether the two summarizations are focusing on the same central topic. Please answer Yes or No."},
    {"role": "user", "content": "1. the Gothic architecture. The Gothic architecture is a style of architecture that flourished in Europe during the High and Late Middle Ages. It evolved from Romanes"},
    {"role": "user", "content": "2. the Gothic architecture. The Gothic architecture is a style of architecture that flourished in Europe during the high and late medieval period. It evolved from Romanes"},
    {"role": "assistant", "content": "Yes."},
    {"role": "user", "content": "You will be given two (potentially truncated) summarizations of a particular text, and your goal is to judge whether the two summarizations are focusing on the same central topic. Please answer Yes or No."},
    {"role": "user", "content": "1. the statue of the civil rights activist, the statue is located in the city of Montgomery, Alabama, the statue is a tribute to the civil rights activist"},
    {"role": "user", "content": "2. the differential geometry of curves and surfaces. The differential geometry of curves and surfaces is a branch of mathematics that studies the geometry of curves and surfaces. The"},
    {"role": "assistant", "content": "No."},
    {"role": "user", "content": "You will be given two (potentially truncated) summarizations of a particular text, and your goal is to judge whether the two summarizations are focusing on the same central topic. Please answer Yes or No."},
    {"role": "user", "content": "1. the artist, and the following is about the art. The artist is a woman, and the art is a painting. The painting is a dream"},
    {"role": "user", "content": "2. the artist’s work, and the following text is about the artist’s life. The artist was born in 1989 in the city of Tiju"},
    {"role": "assistant", "content": "Yes."},
    {"role": "user", "content": "You will be given two (potentially truncated) summarizations of a particular text, and your goal is to judge whether the two summarizations are focusing on the same central topic. Please answer Yes or No."},
    {"role": "user", "content": "1. the influence of ancient Greek tragedies on contemporary theater. The author argues that the archetypal themes, dramatic irony, and complex character development of ancient"},
    {"role": "user", "content": "2. the blockchain, which is a distributed ledger technology that allows for secure, transparent, and tamper-proof transactions. It is a decentralized system that does not"},
    {"role": "assistant", "content": "No."},
]

# responses = []
# for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
#     question = row['Question']
    
try:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=few_shot_messages + [
            {"role": "user", "content": "You will be given two (potentially truncated) summarizations of a particular text, and your goal is to judge whether the two summarizations are focusing on the same central topic. Please answer Yes or No."},
            {"role": "user", "content": f"1. {x1_generated}"},
            {"role": "user", "content": f"2. {x1_adv_generated}"},
        ]
    )
    answer = response.choices[0].message.content
except Exception as e:
    answer = f"Error: {e}"

print(f"LLM Judge Answer: {answer}")

# Optional: avoid hitting rate limits
# time.sleep(1)

# df['response'] = responses
# df.to_csv("./TruthfulQA_rated.csv", index=False)