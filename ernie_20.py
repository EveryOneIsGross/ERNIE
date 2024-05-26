import torch
import re
import string
import random
from transformers import BertTokenizer, BertForMaskedLM, GPT2LMHeadModel, GPT2Tokenizer
from collections import Counter

model_name = "bert-large-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_text(text):
    text = re.sub(r'[^\w\s\[\]]', '', text)  # Allow brackets for [MASK] token
    text = text.replace('[MASK]', tokenizer.mask_token)
    return text

def postprocess_text(text):
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace(tokenizer.mask_token, '[MASK]')  # Convert mask token back to [MASK]
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
    infill_token_ids = sorted_indices[:top_k]
    return infill_token_ids

def score_coherence(sentence, gpt2_model, gpt2_tokenizer, device):
    input_ids = gpt2_tokenizer.encode(sentence, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = gpt2_model(input_ids, labels=input_ids)
        loss = outputs.loss
    tokens = sentence.split()
    token_counts = Counter(tokens)
    repetition_penalty = sum(count ** 2 - 1 for count in token_counts.values() if count > 1) * 0.05
    return -loss.item() - repetition_penalty

def generate_text(seed, gpt2_model, gpt2_tokenizer, max_length=50, num_iterations=20, top_k=5, top_k_depth=3, context_window=5, edge_weight=2.0, punctuation_weight=0.1, mask_probability=0.15):
    seed = preprocess_text(seed)
    seed_tokens = tokenizer.tokenize(seed)
    seed_length = len(seed_tokens)

    print(f"Preprocessed seed: {seed}")
    print(f"Seed tokens: {seed_tokens}")

    generated_tokens = [tokenizer.mask_token] + seed_tokens + [tokenizer.mask_token]
    print(f"Initial generated tokens: {generated_tokens}")

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}:")
        masked_tokens = generated_tokens.copy()
        for i in range(len(masked_tokens)):
            if masked_tokens[i] not in seed_tokens and random.random() < mask_probability:
                masked_tokens[i] = tokenizer.mask_token
        print(f"Masked tokens: {masked_tokens}")

        masked_positions = [i for i, token in enumerate(masked_tokens) if token == tokenizer.mask_token]
        print(f"Masked positions: {masked_positions}")

        if not masked_positions:
            print("No more masked positions. Stopping generation.")
            break

        infill_token_ids_list = []

        for pos in masked_positions:
            left_context = masked_tokens[max(0, pos - context_window):pos]
            right_context = masked_tokens[pos + 1:min(len(masked_tokens), pos + context_window + 1)]
            context_tokens = left_context + [tokenizer.mask_token] + right_context
            context_ids = tokenizer.convert_tokens_to_ids(context_tokens)
            context_ids = torch.tensor([context_ids]).to(device)

            with torch.no_grad():
                outputs = model(context_ids)
                predictions = outputs[0]
                probs = predictions[0, len(left_context)]

            probs[tokenizer.convert_tokens_to_ids(["[UNK]"])[0]] = -float("inf")
            punctuation_ids = set(tokenizer.convert_tokens_to_ids(list(string.punctuation)))
            probs[list(punctuation_ids)] *= punctuation_weight
            probs, indices = torch.topk(probs, k=top_k)
            sorted_probs, sorted_indices, _ = divine_sort(probs, indices)
            infill_token_ids = select_infill_tokens(sorted_probs, sorted_indices, top_k)
            infill_token_ids_list.append(infill_token_ids)

        print(f"Infill token IDs: {infill_token_ids_list}")

        infilled_tokens_list = []
        for combination in torch.cartesian_prod(*infill_token_ids_list):
            infilled_tokens = masked_tokens.copy()
            for i, token_id in enumerate(combination):
                if masked_positions[i] not in range(1, seed_length + 1):
                    infilled_tokens[masked_positions[i]] = tokenizer.convert_ids_to_tokens([token_id.item()])[0]
            infilled_tokens_list.append(infilled_tokens)

        print(f"Infilled tokens list: {infilled_tokens_list}")

        infilled_sentences = [tokenizer.convert_tokens_to_string(tokens) for tokens in infilled_tokens_list]
        print(f"Infilled sentences: {infilled_sentences}")

        ranked_sentences = sorted(infilled_sentences, key=lambda x: score_coherence(x, gpt2_model, gpt2_tokenizer, device), reverse=True)[:top_k_depth]
        print(f"Ranked sentences: {ranked_sentences}")

        generated_tokens = tokenizer.tokenize(ranked_sentences[0])
        generated_tokens = [tokenizer.mask_token] + generated_tokens + [tokenizer.mask_token]
        print(f"Updated generated tokens: {generated_tokens}")


    generated_text = tokenizer.convert_tokens_to_string(generated_tokens)
    generated_text = postprocess_text(generated_text)
    # remove left over [MASK] tokens
    generated_text = generated_text.replace("[MASK]", "")
    
    return generated_text

if __name__ == '__main__':
    gpt2_model_name = "gpt2"
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
    gpt2_model.to(device)

    seed_words = ["goodness"]
    for seed in seed_words:
        print(f"\nSeed: {seed}")
        generated_text = generate_text(seed, gpt2_model, gpt2_tokenizer, max_length=2048, num_iterations=8, top_k=4, top_k_depth=2, context_window=4, edge_weight=1.8, punctuation_weight=0.015, mask_probability=0.15)
        print(f"Generated text: {generated_text}")