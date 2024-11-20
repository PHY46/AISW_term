#%%
# -*- coding: utf-8 -*-
import json
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, EncoderDecoderModel
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, f1_score
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
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
# BPE 토크나이저 정의 및 학습
tokenizer_ko = Tokenizer(models.BPE())
tokenizer_en = Tokenizer(models.BPE())

tokenizer_ko.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer_en.pre_tokenizer = pre_tokenizers.ByteLevel()

tokenizer_ko.decoder = decoders.ByteLevel()
tokenizer_en.decoder = decoders.ByteLevel()

trainer_ko = trainers.BpeTrainer(special_tokens=["<pad>", "<unk>", "<s>", "</s>"])
tokenizer_ko.train_from_iterator(df_data['한국어'].tolist(), trainer=trainer_ko)

trainer_en = trainers.BpeTrainer(special_tokens=["<pad>", "<unk>", "<s>", "</s>"])
tokenizer_en.train_from_iterator(df_data['영어'].tolist(), trainer=trainer_en)

class TextDataset(Dataset):
    def __init__(self, df):
        self.source_ids = []
        self.labels = []
        self.token_type_ids = []

        for _, row in df.iterrows():
            source_encoding = tokenizer_ko.encode(row['한국어'], add_special_tokens=True)
            target_encoding = tokenizer_en.encode(row['영어'], add_special_tokens=True)

            type_ids = [0] * len(source_encoding.ids) + [1] * len(target_encoding.ids)
            self.token_type_ids.append(type_ids)

            combined_tokens = source_encoding.ids + target_encoding.ids
            self.source_ids.append(combined_tokens)
            self.labels.append(row['error'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.source_ids[idx], self.token_type_ids[idx], self.labels[idx]

def collate_fn(batch):
    source_ids = [item[0] for item in batch]
    token_type_ids = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    max_len_source = max(len(x) for x in source_ids)
    source_ids_padded = [torch.nn.functional.pad(torch.tensor(x), (0, max_len_source - len(x)), value=0) for x in source_ids]

    max_len_token_type = max(len(x) for x in token_type_ids)
    token_type_ids_padded = [torch.nn.functional.pad(torch.tensor(x), (0, max_len_token_type - len(x)), value=0) for x in token_type_ids]

    return torch.stack(source_ids_padded), torch.stack(token_type_ids_padded), torch.tensor(labels)

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

epoch_size = 1

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_size)

for epoch in range(epoch_size):
    model.train()
    for source_ids, token_type_ids, labels in train_loader:
        optimizer.zero_grad()
        attention_mask = (source_ids > 0).float()
        output = model(input_ids=source_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for source_ids, token_type_ids, labels in val_loader:
            attention_mask = (source_ids > 0).float()
            output = model(input_ids=source_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
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

# Precision 및 F1 Score 계산
precision = precision_score(test_y_true, test_y_pred, zero_division=1)
f1 = f1_score(test_y_true, test_y_pred, zero_division=1)

print(f'Test Loss={test_loss/len(test_loader)}, Test Accuracy={test_accuracy/len(test_loader)}')
print(f'Precision: {precision:.4f}, F1 Score: {f1:.4f}')
# %%
# 모델 저장
torch.save(model.state_dict(), './translation_model')
# %%
# Encoder-Decoder 모델 로드
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    'bert-base-multilingual-cased', 'bert-base-multilingual-cased'
)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# 데이터셋 수정
class TextCorrectionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.inputs = df['한국어'].tolist()
        self.targets = df['영어'].tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        input_encodings = self.tokenizer(
            input_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        target_encodings = self.tokenizer(
            target_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )

        return {
            "input_ids": input_encodings["input_ids"].squeeze(),
            "attention_mask": input_encodings["attention_mask"].squeeze(),
            "labels": target_encodings["input_ids"].squeeze(),
        }

# 학습 프로세스
train_dataset = TextCorrectionDataset(train_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()

epoch_size = 1

for epoch in range(epoch_size):
    for batch in train_loader:
        output = model(input_ids=source_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 모델 저장
torch.save(model.state_dict(), "./correction_model.pt")

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

# Precision 및 F1 Score 계산
precision = precision_score(test_y_true, test_y_pred, zero_division=1)
f1 = f1_score(test_y_true, test_y_pred, zero_division=1)

print(f'Test Loss={test_loss/len(test_loader)}, Test Accuracy={test_accuracy/len(test_loader)}')
print(f'Precision: {precision:.4f}, F1 Score: {f1:.4f}')
