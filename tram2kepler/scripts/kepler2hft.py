import transformers
import torch
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert = transformers.BertForSequenceClassification.from_pretrained('scibert_single_label_model').to(device).eval()
tokenizer = transformers.BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

