# %%
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BartTokenizer, BartForConditionalGeneration
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
 
# %%
df = pd.DataFrame(columns = ['원문','번역문'])

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
    temp = temp.sample(frac=0.05, random_state=42)
    df = pd.concat([df,temp[['원문','번역문']]])

df.head()

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# 데이터셋 클래스 정의
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, tokenizer, max_length):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        trg = self.trg_sentences[idx]

        # 인코딩
        encoding = self.tokenizer.encode_plus(
            src,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )

        target_encoding = self.tokenizer.encode_plus(
            trg,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()  # labels는 trg로 설정
        }

# 모델과 토크나이저 초기화
model_name = 'facebook/bart-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 하이퍼파라미터 설정
max_length = 64
batch_size = 16
epochs = 5

# 데이터셋 준비
src_sentences = df['원문'].tolist()
trg_sentences = df['번역문'].tolist()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

train_dataset = TranslationDataset(train_df['원문'].tolist(), train_df['번역문'].tolist(), tokenizer, max_length)
val_dataset = TranslationDataset(val_df['원문'].tolist(), val_df['번역문'].tolist(), tokenizer, max_length)
test_dataset = TranslationDataset(test_df['원문'].tolist(), test_df['번역문'].tolist(), tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# 학습 루프
model.to(device)
model.train()
for epoch in range(epochs):
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

            predictions = outputs.logits.argmax(dim=-1)
            val_accuracy += (predictions == labels).float().mean().item()

    print(f'Epoch {epoch + 1}: Val Loss={val_loss/len(val_loader)}, Val Accuracy={val_accuracy/len(val_loader)}')

# 모델 저장
model.save_pretrained('translation_model')
tokenizer.save_pretrained('translation_tokenizer')

# %%
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 저장된 모델과 토크나이저 불러오기
model_path = 'translation_model'
tokenizer_path = 'translation_tokenizer'

model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# BLEU 점수 계산 함수
def calculate_bleu(model, tokenizer, dataset, max_length=64):
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for _, row in tqdm(dataset.iterrows(), desc="Calculating BLEU", total=len(dataset)):
            source_sentence = row['원문']
            target_sentence = row['번역문']

            inputs = tokenizer(source_sentence, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

            output = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=max_length)
            hypothesis = tokenizer.decode(output[0], skip_special_tokens=True)

            references.append([target_sentence.split()])
            hypotheses.append(hypothesis.split())

    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score

# 테스트 데이터셋 불러오기
test_df = pd.read_csv('test_df.csv')
test_df = test_df.sample(frac=0.1, random_state=42)

bleu_score = calculate_bleu(model, tokenizer, test_df)
print(f"BLEU Score: {bleu_score}")
