# -*- coding: utf-8 -*-
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 모델과 토크나이저 로드
translation_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
translation_model.load_state_dict(torch.load('./translation_model', map_location=torch.device('cpu')))
translation_model.eval()

correction_model = AutoModelForSeq2SeqLM.from_pretrained('correction_model')
correction_tokenizer = AutoTokenizer.from_pretrained('correction_tokenizer')

tokenizer_ko = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')

st.title("Sentence Error Checker")

# 사용자 입력 받기
source_sentence = st.text_input("Enter a Korean sentence:", "")
target_sentence = st.text_input("Enter an English sentence:", "")

# 확인 및 결과
if st.button("Check"):
    if source_sentence:
        # 입력 문장 토큰화
        input_ids_ko = tokenizer_ko.encode(source_sentence, return_tensors='pt', truncation=True, padding="max_length", max_length=128)
        input_ids_en = tokenizer_en.encode(target_sentence, return_tensors='pt', truncation=True, padding="max_length", max_length=128)

        combined_input = torch.cat((input_ids_ko, input_ids_en), dim=1)

        # 모델 예측
        with torch.no_grad():
            output = translation_model(combined_input)
            similarity_score = torch.sigmoid(output.logits[:, 1]).item()
            print(similarity_score)

        # 예측 결과
        if similarity_score < 0.3:
            result = "The sentence has no errors."
        else:
            inputs = correction_tokenizer(source_sentence, return_tensors="pt", padding=True, max_length=64, truncation=True)

            with torch.no_grad():

                output = correction_model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                corrected_sentence = correction_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            result = f"The sentence has errors. Suggested correction: {corrected_sentence}"

        st.write(result)

# streamlit run app.py