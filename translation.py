#%%
# -*- coding: utf-8 -*-
import json
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, f1_score
#%%
# 데이터 전처리
df_data = pd.read_excel('ko-eng_data.xlsx')
df_data = df_data.sample(frac=0.1, random_state=42)

df_data = df_data[['한국어', '영어검수']].astype(str)
df_data.columns = ['한국어', '영어']
df_data['error'] = 0

with open('error_data.json', 'r', encoding='utf-8') as f:
    error_data = json.load(f)
df_error = pd.DataFrame(error_data)
df_error.columns = ['한국어', '영어', 'error']
df_error['error'] = 1
# %%
# 데이터셋 분할
data_train_df, data_test_df = train_test_split(df_data, test_size=0.2, random_state=42)
data_train_df, data_val_df = train_test_split(data_train_df, test_size=0.2, random_state=42)

error_train_df, error_test_df = train_test_split(df_error, test_size=0.2, random_state=42)
error_train_df, error_val_df = train_test_split(error_train_df, test_size=0.2, random_state=42)

# 데이터셋 병합
train_df = pd.concat([data_train_df, error_train_df], ignore_index=True)
val_df = pd.concat([data_val_df, error_val_df], ignore_index=True)
test_df = pd.concat([data_test_df, error_test_df], ignore_index=True)
# %%
# 토크나이저 및 임베딩
tokenizer_ko = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')

class TextDataset(Dataset):
    def __init__(self, df):
        self.source_ids = []
        self.target_ids = []
        self.labels = []

        for _, row in df.iterrows():
            source_tokens = tokenizer_ko.encode(row['한국어'], add_special_tokens=True, max_length=128, truncation=True)
            target_tokens = tokenizer_en.encode(row['영어'], add_special_tokens=True, max_length=128, truncation=True)
            self.source_ids.append(source_tokens)
            self.target_ids.append(target_tokens)
            self.labels.append(row['error'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.source_ids[idx], self.target_ids[idx], self.labels[idx]

def collate_fn(batch):
    source_ids = [item[0] for item in batch]
    target_ids = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    max_len = max(len(x) for x in source_ids)
    source_ids_padded = [torch.nn.functional.pad(torch.tensor(x), (0, max_len - len(x)), value=0) for x in source_ids]

    max_len = max(len(x) for x in target_ids)
    target_ids_padded = [torch.nn.functional.pad(torch.tensor(x), (0, max_len - len(x)), value=0) for x in target_ids]

    return torch.stack(source_ids_padded), torch.stack(target_ids_padded), torch.tensor(labels)

# %%
# DataLoader 생성
train_dataset = TextDataset(train_df)
val_dataset = TextDataset(val_df)
test_dataset = TextDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
# %%
# 모델 학습
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')

optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

epoch_size = 2

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_size)

for epoch in range(epoch_size):
    model.train()
    for source_ids, target_ids, labels in train_loader:
        optimizer.zero_grad()
        output = model(input_ids=source_ids, attention_mask=(source_ids > 0).float(), labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for source_ids, target_ids, labels in val_loader:
            output = model(input_ids=source_ids, attention_mask=(source_ids > 0).float(), labels=labels)
            val_loss += output.loss.item()
            val_accuracy += (output.logits.argmax(1) == labels).float().mean().item()

    scheduler.step()

    print(f'Epoch {epoch}: Val Loss={val_loss/len(val_loader)}, Val Accuracy={val_accuracy/len(val_loader)}')

# %%
# 모델 평가
model.eval()
test_loss = 0
test_accuracy = 0
test_y_true = []
test_y_pred = []
with torch.no_grad():
    for source_ids, target_ids, labels in test_loader:
        output = model(input_ids=source_ids, attention_mask=(source_ids > 0).float(), labels=labels)
        test_loss += output.loss.item()
        test_accuracy += (output.logits.argmax(1) == labels).float().mean().item()
        test_y_true.extend(labels.tolist())
        test_y_pred.extend(output.logits.argmax(1).tolist())

precision = precision_score(test_y_true, test_y_pred, zero_division=1)
f1 = f1_score(test_y_true, test_y_pred, zero_division=1)

print(f'Test Loss={test_loss/len(test_loader)}, Test Accuracy={test_accuracy/len(test_loader)}')
print(f'Precision: {precision:.4f}, F1 Score: {f1:.4f}')

# %%
# 모델 저장
torch.save(model.state_dict(), './nlp_model')

# %%
# 모델 불러오기
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn

model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
model.load_state_dict(torch.load('./nlp_model'))
tokenizer_ko = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')
# %%
# 사용자 입력 처리
def check_sentence_similarity(source_sentence, target_sentence):
    source_tokens = tokenizer_ko.encode(source_sentence, add_special_tokens=True, max_length=128, truncation=True)
    target_tokens = tokenizer_en.encode(target_sentence, add_special_tokens=True, max_length=128, truncation=True)
    output = model(input_ids=torch.tensor([source_tokens]), attention_mask=(torch.tensor([source_tokens]) > 0).float(), labels=torch.tensor([0]))
    similarity_score = nn.functional.softmax(output.logits, dim=1)[0][0].item()
    #print(similarity_score)
    if similarity_score > 0.5:
        return "The sentence has no errors."
    else:
        return "The sentence has errors."
# %%
# 사용자 입력 받기
#source_sentence = input("한국어 문장을 입력하세요: ")
#target_sentence = input("영어 문장을 입력하세요: ")

#check_sentence_similarity(source_sentence, target_sentence)
# %%
# -*- coding: utf-8 -*-
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn

st.title("Sentence Similarity Checker")

# Get user input
source_sentence = st.text_area("Enter a Korean sentence:", "")

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Check sentence similarity and display the result
if st.button("Check"):
    if source_sentence:
        # Tokenize the input sentence
        input_ids = tokenizer.encode(source_sentence, return_tensors='pt')

        # Pass the input through the model
        output = model(input_ids)[0]

        # Get the similarity score
        similarity_score = nn.functional.softmax(output, dim=1)[0][0].item()

        if similarity_score > 0.5:
            result = "The sentence has no errors."
        else:
            result = "The sentence has errors."

        st.write(result)


# %%
#streamlit run translation.py
# %%