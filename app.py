# -*- coding: utf-8 -*-
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 모델과 토크나이저 로드
translation_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
translation_model.load_state_dict(torch.load('./translation_model', map_location=torch.device('cpu')))
translation_model.eval()

correction_model = torch.load('./correction_model.pt', map_location=torch.device('cpu'))
correction_model.eval()

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
            with torch.no_grad():
                encoder_output = correction_model.encoder(input_ids_ko)
                hidden_states = encoder_output.last_hidden_state
                decoder_input_ids = tokenizer_ko.encode(tokenizer_ko.cls_token, return_tensors='pt').to(hidden_states.device)  # cls_token 사용

                output_ids = correction_model.decoder.generate(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=hidden_states,
                    max_new_tokens=50,
                    num_beams=5,
                    early_stopping=True
                )
                corrected_sentence = tokenizer_ko.decode(output_ids[0], skip_special_tokens=True)

            result = f"The sentence has errors. Suggested correction: {corrected_sentence}"

        st.write(result)

# streamlit run app.py