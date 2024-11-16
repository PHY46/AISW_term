from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
model.load_state_dict(torch.load('./nlp_model'))
tokenizer_ko = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')

# User input processing
def check_sentence_similarity(source_sentence, target_sentence):
    source_tokens = tokenizer_ko.encode(source_sentence, add_special_tokens=True, max_length=256, truncation=True)
    target_tokens = tokenizer_en.encode(target_sentence, add_special_tokens=True, max_length=256, truncation=True)

    model_name = "bert-base-multilingual-cased"
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.load_state_dict(torch.load('nlp_model'))

    output = model(input_ids=torch.tensor([source_tokens]), attention_mask=(torch.tensor([source_tokens]) > 0).float())
    similarity_score = nn.functional.softmax(output.logits, dim=1)[0][0].item()

    if similarity_score > 0.5:
        return "The sentence has no errors."
    else:
        return "The sentence has errors."

# Streamlit application setup
import streamlit as st

st.title("Sentence Similarity Checker")

# Get user input
source_sentence = st.text_input("Enter a Korean sentence:", "")
target_sentence = st.text_input("Enter an English sentence:", "")

# Check sentence similarity and display the result
if st.button("Check"):
    result = check_sentence_similarity(source_sentence, target_sentence)
    st.write(result)

# streamlit run app.py --server.port 20000