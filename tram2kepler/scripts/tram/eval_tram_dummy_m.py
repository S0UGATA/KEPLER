import os
import pickle
import random

print(os.environ['CONDA_DEFAULT_ENV'])

from sklearn.preprocessing import MultiLabelBinarizer as MLB

CLASSES = [
    'T1003.001', 'T1005', 'T1012', 'T1016', 'T1021.001', 'T1027',
    'T1033', 'T1036.005', 'T1041', 'T1047', 'T1053.005', 'T1055',
    'T1056.001', 'T1057', 'T1059.003', 'T1068', 'T1070.004',
    'T1071.001', 'T1072', 'T1074.001', 'T1078', 'T1082', 'T1083',
    'T1090', 'T1095', 'T1105', 'T1106', 'T1110', 'T1112', 'T1113',
    'T1140', 'T1190', 'T1204.002', 'T1210', 'T1218.011', 'T1219',
    'T1484.001', 'T1518.001', 'T1543.003', 'T1547.001', 'T1548.002',
    'T1552.001', 'T1557.001', 'T1562.001', 'T1564.001', 'T1566.001',
    'T1569.002', 'T1570', 'T1573.001', 'T1574.002'
]

mlb = MLB(classes=CLASSES)
mlb.fit([[c] for c in CLASSES])

print(mlb)

import pandas as pd

# data = pd.read_json('/home/sougata/projects/MyKEPLER/tram2kepler/data/input/multi_label.json').drop(
#     columns='doc_title')
# print(data.head(10))

# In[ ]:


import transformers
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert = transformers.BertForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-BertModel", num_labels=50).to(device).eval()
tokenizer = transformers.BertTokenizer.from_pretrained("hf-internal-testing/tiny-random-BertModel")

print(bert)

from sklearn.model_selection import train_test_split

random.seed(42)
with open('train_test_data_m.pkl', 'rb') as f:
    train, test = pickle.load(f)


def _load_data(x, y, batch_size=10):
    x_len, y_len = x.shape[0], y.shape[0]
    assert x_len == y_len
    for i in range(0, x_len, batch_size):
        slc = slice(i, i + batch_size)
        yield x[slc].to(device), y[slc].to(device)


def _tokenize(instances: list[str]):
    return tokenizer(instances, return_tensors='pt', padding='max_length', truncation=True, max_length=512).input_ids


def _encode_labels(labels):
    """:labels: should be the `labels` column (a Series) of the DataFrame"""
    return torch.Tensor(mlb.transform(labels.to_numpy()))


# In[6]:
bert.eval()

x_train = _tokenize(train['sentence'].tolist())
print(x_train)

y_train = _encode_labels(train['labels'])
print(y_train)
print(y_train.sum())

x_test = _tokenize(test['sentence'].tolist())
y_test = _encode_labels(test['labels'])

batch_size = 20
preds = []

with torch.no_grad():
    for i in range(0, x_test.shape[0], batch_size):
        x = x_test[i: i + batch_size].to(device)
        out = bert(x, attention_mask=x.ne(tokenizer.pad_token_id).to(int))
        preds.extend(out.logits.to('cpu'))

binary_preds = torch.vstack(preds).sigmoid().gt(.5).to(int)

preds_series = pd.Series(mlb.inverse_transform(binary_preds)).apply(frozenset)
actual_series = pd.Series(mlb.inverse_transform(y_test)).apply(frozenset)
results = pd.concat({'predicted': preds_series, 'actual': actual_series}, axis=1)

print(results)

# In[11]:


tp = results.apply((lambda r: r.predicted & r.actual), axis=1).explode().value_counts()
fp = results.apply((lambda r: r.predicted - r.actual), axis=1).explode().value_counts()
fn = results.apply((lambda r: r.actual - r.predicted), axis=1).explode().value_counts()
counts = pd.concat({'tp': tp, 'fp': fp, 'fn': fn}, axis=1).fillna(0).astype(int)

support = actual_series.explode().value_counts().rename('#')

p = counts.tp.div(counts.tp + counts.fp).fillna(0)
r = counts.tp.div(counts.tp + counts.fn).fillna(0)
f1 = (2 * p * r) / (p + r)

scores = pd.concat({'P': p, 'R': r, 'F1': f1}, axis=1).fillna(0).sort_values(by='F1', ascending=False)

# calculate macro scores
scores.loc['(macro)'] = scores.mean()

# calculate micro scores
micro = counts.sum()
scores.loc['(micro)', 'P'] = mP = micro.tp / (micro.tp + micro.fp)
scores.loc['(micro)', 'R'] = mR = micro.tp / (micro.tp + micro.fn)
scores.loc['(micro)', 'F1'] = (2 * mP * mR) / (mP + mR)

print(scores.join(support))
