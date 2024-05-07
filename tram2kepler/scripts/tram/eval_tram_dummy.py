import os
import random

print(os.environ['CONDA_DEFAULT_ENV'])

# This cell instantiates the label encoder. Do not modify this cell, as the classes (ie, ATT&CK techniques) and their order must match those the model expects.

# In[2]:


from sklearn.preprocessing import OneHotEncoder as OHE

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

encoder = OHE(sparse_output=False)
encoder.fit([[c] for c in CLASSES])

print(encoder.categories_)

# This cell is for loading the training data. You will need to modify this cell to load your data. Ensure that by the end of this cell, a DataFrame has been assigned to the variable `data` that has a `text` column containing the segments, and a `label` column containing individual strings, where those strings are an ATT&CK IDs that this model can classify. It does not matter how the DataFrame is indexed or what other columns with other names, if any, it has.
# 
# For demonstration purposes, we will use the same single-label data that was produced during this TRAM effort, even though the model was trained on this data already. This cell is only present to show the expected format of the `data` DataFrame, and is not intended to be run as shown.

# In[3]:


import pandas as pd

data = pd.read_json('/home/sougata/projects/MyKEPLER/tram2kepler/data/input/single_label.json').drop(
    columns='doc_title')
print(data.head(10))

# In[4]:


import transformers
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert = transformers.BertForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-BertModel", num_labels=50).to(device).eval()
tokenizer = transformers.BertTokenizer.from_pretrained("hf-internal-testing/tiny-random-BertModel")

print(bert)

from sklearn.model_selection import train_test_split

random.seed(42)
train, test = train_test_split(data, test_size=0.2, shuffle=True)


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
    return torch.Tensor(encoder.transform(labels))

bert.eval()

x_train = _tokenize(train['text'].tolist())
print(x_train)

y_train = _encode_labels(train[['label']])
print(y_train)
print(y_train.sum())

x_test = _tokenize(test['text'].tolist())
y_test = test['label']
y_test_enc = _encode_labels(test[['label']])

batch_size = 20
preds = []

with torch.no_grad():
    for i in range(0, x_test.shape[0], batch_size):
        x = x_test[i: i + batch_size].to(device)
        out = bert(x, attention_mask=x.ne(tokenizer.pad_token_id).to(int))
        preds.extend(out.logits.to('cpu'))

import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support as calculate_score

predicted_labels = (
    encoder.inverse_transform(
        F.one_hot(
            torch.vstack(preds).softmax(-1).argmax(-1),
            num_classes=len(encoder.categories_[0])
        )
        .numpy()
    )
    .reshape(-1)
)

predicted = list(predicted_labels)
actual = y_test.tolist()

labels = sorted(set(actual) | set(predicted))

scores = calculate_score(actual, predicted, labels=labels)

scores_df = pd.DataFrame(scores).T
scores_df.columns = ['P', 'R', 'F1', '#']
scores_df.index = labels
scores_df.loc['(micro)'] = calculate_score(actual, predicted, average='micro', labels=labels)
scores_df.loc['(macro)'] = calculate_score(actual, predicted, average='macro', labels=labels)

print(scores_df)
