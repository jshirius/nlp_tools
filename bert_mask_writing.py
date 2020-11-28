#!/usr/bin/env python
# coding: utf-8


import torch
from transformers.modeling_bert import BertForMaskedLM
from transformers.tokenization_bert_japanese import  BertJapaneseTokenizer
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelWithLMHead

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-i', '--input_string', type=str,
                           default="",
                           help='文章を入力します。予測させたい単語はアスターリスク')
    return argparser.parse_args()
    
    
args = get_option()
print(str(args))

tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
model = AutoModelWithLMHead.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
model.eval()


text = args.input_string

tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)

#文の先頭に[CLS]を追加する
tokenized_text.insert(0, '[CLS]')

#「*」をmaskする(予測させる単語)
tokenized_text = [i.replace("*","[MASK]") for i in tokenized_text]

#文章の区切りを[SEP]を設定する
tokenized_text = [i.replace("。","[SEP]") for i in tokenized_text]

masked_index_list = []
for index, text_data in enumerate(tokenized_text):
    if("[MASK]" == text_data):
            masked_index_list.append(index)
        

print(tokenized_text)
# Convert token to vocabulary indices

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

print(indexed_tokens)
tokens_tensor = torch.tensor([indexed_tokens])


# Predict
predictions_list = []
with torch.no_grad():
    outputs = model(tokens_tensor)
    #print(outputs[0][0, masked_index])
    for masked_index in masked_index_list:
        predictions = outputs[0][0, masked_index].topk(9) # 予測結果の上位9件を抽出
        predictions_list.append(predictions)

# Show results
token_list = []
for predictions in predictions_list:
    print("予測結果の出力")
    for i, index_t in enumerate(predictions.indices):
        index = index_t.item()
        token = tokenizer.convert_ids_to_tokens([index])[0]
        print(i, token)
        token_list.append(token)

