# encoding: utf-8
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging


class NextTokenPredictor(object):

    def __init__(self, model_name='bert'):
        pass


def test_example(model_name='bert-base-chinese'):

    tokenizer = BertTokenizer.from_pretrained(model_name)
    text = "你也在杭州工作吗？"
    tokenized_txt = tokenizer.tokenize(text)
    tokenized_txt = ['[CLS]'] + tokenized_txt + ['[SEP]']
    print(tokenized_txt)
    mask_index = 3
    real_word = tokenized_txt[mask_index]
    tokenized_txt[mask_index] = '[MASK]'
    print(tokenized_txt)
    segment_ids = [0]*len(tokenized_txt)
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_txt)
    print(segment_ids)
    print(token_ids)
    token_tensor = torch.tensor(token_ids)
    seg_tensor = torch.tensor(segment_ids)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.eval()

    # If you have a GPU, put everything on cuda
    device_name = 'cpu'
    if torch.cuda.is_available():
        device_name = 'cuda'
    tokens_tensor = token_tensor.to(device_name)
    segments_tensors = seg_tensor.to(device_name)
    model.to(device_name)

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0]

    # confirm we were able to predict 'henson'
    predicted_index = torch.argmax(predictions[0, mask_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token)


if __name__ == "__main__":
    test_example()
