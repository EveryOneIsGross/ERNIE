from transformers import BertTokenizer, BertForMaskedLM, GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
from itertools import product
import random

# Load models and tokenizers
model_name = "bert-large-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)
gpt2_model_name = "gpt2"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
gpt2_model.to(device)

def preprocess_text(text):
    return text.replace('[MASK]', tokenizer.mask_token)

def postprocess_text(text):
    text = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def divine_sort(probs, indices, threshold=0.5):
    n = len(probs)
    moves = 0
    for i in range(n // 2):
        max_index = i
        for j in range(i + 1, n - i):
            if probs[j] > probs[max_index]:
                max_index = j
        if max_index != i:
            probs[i], probs[max_index] = probs[max_index], probs[i]
            indices[i], indices[max_index] = indices[max_index], indices[i]
            moves += 1
        if probs[i] >= threshold:
            break
    return probs, indices, moves

def select_infill_tokens(sorted_probs, sorted_indices, top_k):
    return sorted_indices[:top_k]

def score_coherence(sentence):
    input_ids = gpt2_tokenizer.encode(sentence, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = gpt2_model(input_ids, labels=input_ids)
        loss = outputs.loss
    return -loss.item()

def infill_text(text, max_iterations=10, context_window=5, top_k=10, top_k_depth=5):
    text = preprocess_text(text)
    tokens = tokenizer.tokenize(text)
    masked_positions = [i for i, token in enumerate(tokens) if token == tokenizer.mask_token]
    
    infilled_sentences = []
    current_text = tokens.copy()

    print(f"Initial text: {tokenizer.convert_tokens_to_string(current_text)}")
    print(f"Masked positions: {masked_positions}")

    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}:")
        infill_token_ids_list = []
        for pos in masked_positions:
            left_context = current_text[max(0, pos - context_window):pos]
            right_context = current_text[pos + 1:min(len(current_text), pos + context_window + 1)]
            context_ids = tokenizer.convert_tokens_to_ids(left_context + [tokenizer.mask_token] + right_context)
            context_ids = torch.tensor([context_ids]).to(device)

            with torch.no_grad():
                outputs = model(context_ids)
                predictions = outputs[0]
                probs = predictions[0, len(left_context)]
                probs, indices = torch.topk(probs, k=top_k)  # Sample from the top-k distribution

            sorted_probs, sorted_indices, _ = divine_sort(probs, indices)
            infill_token_ids = select_infill_tokens(sorted_probs, sorted_indices, top_k)
            infill_token_ids_list.append(infill_token_ids)

        # Generate possible infilled texts and select the best based on coherence
        for combination in product(*infill_token_ids_list):
            infilled_tokens = current_text.copy()
            for i, token_id in enumerate(combination):
                infilled_tokens[masked_positions[i]] = tokenizer.convert_ids_to_tokens([token_id])[0]

            infilled_text = tokenizer.convert_tokens_to_string(infilled_tokens)
            infilled_text = postprocess_text(infilled_text)
            if infilled_text not in infilled_sentences:
                infilled_sentences.append(infilled_text)

        print(f"Generated sentences: {infilled_sentences}")

        # Re-mask based on low confidence and refine
        current_text = tokenizer.tokenize(infilled_sentences[0])  # Start with the best from the last round
        masked_positions = [i for i, token in enumerate(current_text) if random.random() < 0.15]  # Randomly re-mask 15% of the tokens
        text = tokenizer.convert_tokens_to_string(current_text)
        text = preprocess_text(text)  # Apply masking

        print(f"Re-masked text: {text}")
        print(f"New masked positions: {masked_positions}")

    ranked_sentences = sorted(infilled_sentences, key=score_coherence, reverse=True)[:top_k_depth]
    return ranked_sentences

# Example usage
seed_text = "[MASK] good  was [MASK] over the [MASK] dog [MASK]."
print(f"Seed: {seed_text}")
infilled_sentences = infill_text(seed_text, max_iterations=3, context_window=8, top_k=4, top_k_depth=4)
for i, sentence in enumerate(infilled_sentences, start=1):
    print(f"Top-{i}: {sentence}")


    '''needs punctuation_weight, edge_weight, and score_coherence function'''