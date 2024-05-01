#!/usr/bin/env python
# coding: utf-8

# # TRAM-LLM, Multi-label
# 
# [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/center-for-threat-informed-defense/tram/blob/main/user_notebooks/predict_multi_label.ipynb)
# 
# This notebook allows one to apply the multi-label SciBERT model for TRAM.
# 
# To start, first select `Runtime > Change runtime type`, and under `Hardware accelerator` select `GPU`. Then run the next two cells. The first cell will download the model and the Python dependencies. The second cell will load the model and set up the selectors.

# In[ ]:


get_ipython().system('mkdir scibert_multi_label_model')
get_ipython().system('wget https://ctidtram.blob.core.windows.net/tram-models/multi-label-20230803/config.json -O scibert_multi_label_model/config.json')
get_ipython().system('wget https://ctidtram.blob.core.windows.net/tram-models/multi-label-20230803/pytorch_model.bin -O scibert_multi_label_model/pytorch_model.bin')
get_ipython().system('pip install torch transformers pandas python-docx pdfplumber bs4')


# In[ ]:


import transformers
import torch
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert = transformers.BertForSequenceClassification.from_pretrained('scibert_multi_label_model').to(device).eval()
tokenizer = transformers.BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

import pandas as pd
from tqdm import tqdm

CLASSES = (
    'T1003.001', 'T1005', 'T1012', 'T1016', 'T1021.001', 'T1027',
    'T1033', 'T1036.005', 'T1041', 'T1047', 'T1053.005', 'T1055',
    'T1056.001', 'T1057', 'T1059.003', 'T1068', 'T1070.004',
    'T1071.001', 'T1072', 'T1074.001', 'T1078', 'T1082', 'T1083',
    'T1090', 'T1095', 'T1105', 'T1106', 'T1110', 'T1112', 'T1113',
    'T1140', 'T1190', 'T1204.002', 'T1210', 'T1218.011', 'T1219',
    'T1484.001', 'T1518.001', 'T1543.003', 'T1547.001', 'T1548.002',
    'T1552.001', 'T1557.001', 'T1562.001', 'T1564.001', 'T1566.001',
    'T1569.002', 'T1570', 'T1573.001', 'T1574.002'
)

ID_TO_NAME = {"T1055": "Process Injection", "T1110": "Brute Force", "T1055.004": "Asynchronous Procedure Call", "T1047": "Windows Management Instrumentation", "T1078": "Valid Accounts", "T1140": "Deobfuscate/Decode Files or Information", "T1016": "System Network Configuration Discovery", "T1057": "Process Discovery", "T1078.004": "Cloud Accounts", "T1518.001": "Security Software Discovery", "T1090.001": "Internal Proxy", "T1078.001": "Default Accounts", "T1071.001": "Web Protocols", "T1082": "System Information Discovery", "T1110.003": "Password Spraying", "T1484.001": "Group Policy Modification", "T1106": "Native API", "T1027.008": "Stripped Payloads", "T1548.002": "Bypass User Account Control", "T1105": "Ingress Tool Transfer", "T1033": "System Owner/User Discovery", "T1569.002": "Service Execution", "T1566.001": "Spearphishing Attachment", "T1059.003": "Windows Command Shell", "T1053.005": "Scheduled Task", "T1547.001": "Registry Run Keys / Startup Folder", "T1041": "Exfiltration Over C2 Channel", "T1210": "Exploitation of Remote Services", "T1005": "Data from Local System", "T1219": "Remote Access Software", "T1552.001": "Credentials In Files", "T1068": "Exploitation for Privilege Escalation", "T1543.003": "Windows Service", "T1570": "Lateral Tool Transfer", "T1027": "Obfuscated Files or Information", "T1113": "Screen Capture", "T1078.003": "Local Accounts", "T1012": "Query Registry", "T1055.002": "Portable Executable Injection", "T1573.001": "Symmetric Cryptography", "T1055.001": "Dynamic-link Library Injection", "T1072": "Software Deployment Tools", "T1027.001": "Binary Padding", "T1190": "Exploit Public-Facing Application", "T1218.011": "Rundll32", "T1090.003": "Multi-hop Proxy", "T1055.012": "Process Hollowing", "T1056.001": "Keylogging", "T1055.008": "Ptrace System Calls", "T1204.002": "Malicious File", "T1083": "File and Directory Discovery", "T1070.004": "File Deletion", "T1110.004": "Credential Stuffing", "T1036.005": "Match Legitimate Name or Location", "T1574.002": "DLL Side-Loading", "T1090": "Proxy", "T1027.003": "Steganography", "T1027.007": "Dynamic API Resolution", "T1074.001": "Local Data Staging", "T1090.002": "External Proxy", "T1564.001": "Hidden Files and Directories", "T1021.001": "Remote Desktop Protocol", "T1112": "Modify Registry", "T1027.005": "Indicator Removal from Tools", "T1003.001": "LSASS Memory", "T1027.002": "Software Packing", "T1090.004": "Domain Fronting", "T1562.001": "Disable or Modify Tools", "T1027.006": "HTML Smuggling", "T1095": "Non-Application Layer Protocol", "T1027.009": "Embedded Payloads", "T1078.002": "Domain Accounts"}

def create_subsequences(document: str, n: int = 13, stride: int = 5) -> list[str]:
    words = document.split()
    subsequences = [' '.join(words[i:i+n]) for i in range(0, len(words), stride)]
    return subsequences

def predict_document(document: str, threshold: float = 0.5, n: int = 13, stride: int = 5):
    text_instances = create_subsequences(document, n, stride)
    tokenized_instances = tokenizer(text_instances, return_tensors='pt', padding='max_length', truncation=True, max_length=512).input_ids

    predictions = []
    batch_size = 10
    slice_starts = tqdm(list(range(0, tokenized_instances.shape[0], batch_size)))

    with torch.no_grad():
        for i in slice_starts:
            x = tokenized_instances[i : i + batch_size].to(device)
            out = bert(x, attention_mask=x.ne(tokenizer.pad_token_id).to(int))
            predictions.extend(out.logits.sigmoid().to('cpu'))

    probabilities = pd.DataFrame(
        torch.vstack(predictions),
        columns=CLASSES,
        index=text_instances
    )

    result: list[tuple[str, set[str]]] = [
        (text, {ID_TO_NAME[k] + ' - ' + k for k, v in clses.items() if v})
        for text, clses in
        probabilities.gt(threshold).T.to_dict().items()
    ]

    result_iter = iter(result)
    current_text, current_labels = next(result_iter)
    overlap = n_selector.value - stride_selector.value
    out = []

    for text, labels in result_iter:
        if labels != current_labels:
            out.append((current_text, current_labels))
            current_text = text
            current_labels = labels
            continue
        current_text += ' ' + ' '.join(text.split()[overlap:])

    out_df = pd.DataFrame(out)
    out_df.columns = ['segment', 'label(s)']
    return out_df

import io
import re
import pdfplumber
import docx
from bs4 import BeautifulSoup

def parse_text(file_name: str, content: io.BytesIO) -> str:
    if file_name.endswith('.pdf'):
        with pdfplumber.open(content) as pdf:
            text = " ".join(page.extract_text() for page in pdf.pages)
    elif file_name.endswith('.html'):
        text = BeautifulSoup(content.read().decode('utf-8'), features="html.parser").get_text()
    elif file_name.endswith('.txt'):
        text = content.read().decode('utf-8')
    elif file_name.endswith('.docx'):
        text = " ".join(paragraph.text for paragraph in docx.Document(content).paragraphs)

    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

from ipywidgets import FileUpload
from IPython.display import display

upload = FileUpload(multiple=True)

from itertools import count
COUNT = count(1)

import ipywidgets as widgets
n_selector = widgets.BoundedIntText(value=13, min=0, step=1, description='n value:', disabled=False)
stride_selector = widgets.BoundedIntText(value=5, min=0, step=1, description='stride size:', disabled=False)
threshold_selector = widgets.BoundedFloatText(value=0.9, min=0, step=0.1, description='probability:', disabled=False)
display(upload, n_selector, stride_selector, threshold_selector)


# Use the above button to select one or more PDF, HTML, Word, or txt files to upload.
# 
# You can use the default values for n, the stride size, and the probability threshold, or set your own.
# 
# - The **n value** is the number of words to include in each segment.
# - The **stride size** is the number of words apart each ngram should start. This needs to be less than the n value, or some words will be skipped
# - The **probability** is the threshold for the model. Setting a lower probability means getting more predictions, but with a lower level of confidence. If the threshold is less than 0.5, you can potentially get two predictions (or three if it's less than 0.33, etc.).
# 
# When you have uploaded the files and selected the parameters, run the next cell to extract text from the files, create the ngrams, and apply the model. The results will be written to the file indicated by `output_file_name`, which you can modify.

# In[5]:


dfs = []
for name, content in zip(upload.value, upload.data):
    text = parse_text(name, io.BytesIO(content))
    prediction_df = predict_document(text, threshold_selector.value, n_selector.value, stride_selector.value)
    prediction_df['name'] = name
    dfs.append(prediction_df)

predicted = pd.concat(dfs).reset_index(drop=True)
i = next(COUNT)
output_file_name = f"./output-{i}.json"
predicted.to_json(output_file_name, orient='table')

predicted

