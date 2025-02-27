# %%
import json
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, f1_score
from tqdm import tqdm

# %%
# 데이터 전처리
df = pd.DataFrame(columns=['원문', '번역문'])

file_list = [
    'data/1_구어체(1).xlsx',
    'data/1_구어체(2).xlsx',
    'data/2_대화체.xlsx',
    'data/3_문어체_뉴스(2).xlsx',
    'data/3_문어체_뉴스(3).xlsx',
    'data/4_문어체_한국문화.xlsx',
]

for data in file_list:
    temp = pd.read_excel(data)
    temp = temp.sample(n=400, random_state=42)
    df_data = pd.concat([df, temp[['원문', '번역문']]])

df_data.columns = ['한국어', '영어']
df_data['error'] = 0

with open('data/error_data.json', 'r', encoding='utf-8') as f:
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
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# 데이터셋 정의
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.inputs = []
        self.labels = []

        for _, row in df.iterrows():
            encoded = tokenizer(
                text=row['한국어'],
                text_pair=row['영어'],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            self.inputs.append(encoded)
            self.labels.append(row['error'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val.squeeze(0) for key, val in self.inputs[idx].items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# %%
# DataLoader 생성
train_dataset = TextDataset(train_df, tokenizer)
val_dataset = TextDataset(val_df, tokenizer)
test_dataset = TextDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# %%
# 모델 학습
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

epoch_size = 5

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

for epoch in range(epoch_size):
    model.train()
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epoch_size}", leave=False):
        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epoch_size}", leave=False):
            output = model(**batch)
            val_loss += output.loss.item()
            val_accuracy += (output.logits.argmax(1) == batch['labels']).float().mean().item()

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
    for batch in test_loader:
        output = model(**batch)
        test_loss += output.loss.item()
        test_accuracy += (output.logits.argmax(1) == batch['labels']).float().mean().item()
        test_y_true.extend(batch['labels'].tolist())
        test_y_pred.extend(output.logits.argmax(1).tolist())

# Precision 및 F1 Score 계산
precision = precision_score(test_y_true, test_y_pred, zero_division=1)
f1 = f1_score(test_y_true, test_y_pred, zero_division=1)

print(f'Test Loss={test_loss/len(test_loader)}, Test Accuracy={test_accuracy/len(test_loader)}')
print(f'Precision: {precision:.4f}, F1 Score: {f1:.4f}')

# %%
# 모델 저장
torch.save(model.state_dict(), './classification_model')
