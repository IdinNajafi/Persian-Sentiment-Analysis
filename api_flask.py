import numpy as np
import torch
from torch import nn
from flask import Flask, jsonify, request
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from config import dropout, max_senten_len
from persian_normal import persian_pre_process


class BertClassifier(nn.Module):
    def __init__(self, model_name):
        super(BertClassifier, self).__init__()
        D_in, D_out = 768, 2
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(D_in, D_out)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

    def segment_forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        mean_lat_hidden = torch.mean(last_hidden_state_cls, dim=0)
        logits = self.classifier(mean_lat_hidden)
        return logits


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
app = Flask(__name__)
model_name = 'HooshvareLab/bert-fa-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
bert_classifier = BertClassifier(model_name)
bert_classifier.load_state_dict(torch.load('Idin-Bert'))
bert_classifier.to(device)
bert_classifier.eval()


def analyze_text(text):
    encoded_sent = tokenizer.encode_plus(
        text=persian_pre_process(text),  # Preprocess sentence
        add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
        max_length=max_senten_len,  # Max length to truncate/pad
        pad_to_max_length=True,  # Pad sentence to max length
        truncation=True,
        return_attention_mask=True  # Return attention mask
    )
    input_id = encoded_sent.get('input_ids')
    attention_mask = encoded_sent.get('attention_mask')
    input_id = torch.tensor(input_id)
    attention_mask = torch.tensor(attention_mask)
    input_id = input_id.to(device)
    attention_mask = attention_mask.to(device)
    probs = bert_classifier(input_id, attention_mask)
    probs = F.softmax(probs, dim=1).detach().cpu().numpy().tolist()
    return probs[0]


def predict_text(text):
    probs = analyze_text(text)
    return np.argmax(probs)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            text = request.form['text']
            if not isinstance(text, str):
                return jsonify('Input must be text!')
            if len(text) == 0:
                return jsonify('empty string!')
            probs = analyze_text(text)
            answer_dict = {'negative': probs[0], 'positive': probs[1]}
            return jsonify(answer_dict)
        except Exception as e:
            return jsonify(str(e))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
