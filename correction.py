# %%
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import EncoderDecoderModel, BertTokenizer
from nltk.translate.bleu_score import sentence_bleu
# %%
# 데이터 전처리
df_data = pd.read_excel('ko-eng_data.xlsx')
df_data = df_data.sample(frac=0.1, random_state=42)

df_data = df_data[['한국어', '영어검수']].astype(str)
df_data.columns = ['한국어', '영어']

# 데이터셋 분할
train_df, test_df = train_test_split(df_data, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
# %%
# Encoder-Decoder 모델 로드
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    'bert-base-multilingual-cased', 'bert-base-multilingual-cased'
)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id

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
        
train_dataset = TextCorrectionDataset(train_df, tokenizer)
val_dataset = TextCorrectionDataset(val_df, tokenizer)
test_dataset = TextCorrectionDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# %%
print(len(train_loader))
# %%
# 모델 학습
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()

epoch_size = 5

for epoch in range(epoch_size):
    for batch in train_loader:
        source_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        output = model(input_ids=source_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()  # Validation 시작
    val_loss = 0
    val_accuracy = 0

    with torch.no_grad():
        for batch in val_loader:
            source_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            output = model(input_ids=source_ids, attention_mask=attention_mask, labels=labels)
            val_loss += output.loss.item()
            val_accuracy += (output.logits.argmax(-1) == labels).float().mean().item()

    print(f'Epoch {epoch}: Val Loss={val_loss/len(val_loader)}, Val Accuracy={val_accuracy/len(val_loader)}')

# %%
# 모델 저장
torch.save(model, "./correction_model.pt")

# %%
# 모델 평가
model.eval()
test_loss = 0
test_accuracy = 0
test_y_true = []
test_y_pred = []
with torch.no_grad():
    for batch in test_loader:
        source_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        attention_mask = (source_ids > 0).float() 
        output = model(input_ids=source_ids, attention_mask=attention_mask, labels=labels)
        test_loss += output.loss.item()
        test_accuracy += (output.logits.argmax(-1) == labels).float().mean().item()
        test_y_true.extend(labels.tolist())
        preds = output.logits.argmax(-1).tolist()
        test_y_pred.extend(preds)

print(f'Test Loss={test_loss/len(test_loader)}, Test Accuracy={test_accuracy/len(test_loader)}')

# BLEU Score 계산
bleu_scores = []

for reference, candidate in zip(test_y_true, test_y_pred):
    score = sentence_bleu([reference], candidate)
    bleu_scores.append(score)

average_bleu = np.mean(bleu_scores)

print(f'Average BLEU Score: {average_bleu:.4f}')
