from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json 


def aggregate_attention(attn):
    '''Extract average attention vector'''
    avged = []
    for layer in attn:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat((
            # We zero the first entry because it's what's called
            # null attention (https://aclanthology.org/W19-4808.pdf)
            torch.tensor([0.]),
            # usually there's only one item in attns_per_head but
            # on the first generation, there's a row for each token
            # in the prompt as well, so take [-1]
            attns_per_head[-1][1:],
            # add zero for the final generated token, which never
            # gets any attention
            torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)

def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])

def decode(tokens, tokenizer):
    '''Turn tokens into text with mapping index'''
    full_text = ''
    chunks = []
    for i, token in enumerate(tokens):
        text = tokenizer.decode(token)
        full_text += text
        chunks.append(text)
    return full_text, chunks

def get_completion(prompt, tokenizer, model):
    '''Get full text, token mapping, and attention matrix for a completion'''
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        tokens,
        max_new_tokens=50,
        num_beams = 1,
        output_attentions=True,
        return_dict_in_generate=True,
        early_stopping=False,
        length_penalty=-1
    )
    sequences = outputs.sequences
    attn_m = heterogenous_stack([
        torch.tensor([
            1 if i == j else 0
            for j, token in enumerate(tokens[0])
        ])
        for i, token in enumerate(tokens[0])
    ] + list(map(aggregate_attention, outputs.attentions)))
    decoded, tokenized = decode(sequences[0], tokenizer)
    return decoded, tokenized, attn_m

def show_matrix(xs):
    for x in xs:
        line = ''
        for y in x:
            line += '{:.4f}\t'.format(float(y))
        print(line)

def attention_view(sparse, filename):
    indices, values = sparse.indices(), sparse.values()
    data = {
        'tokens': tokenized,
        'attn_indices': indices.T.numpy().tolist(),
        'attn_values': values.numpy().tolist(),
    }
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)


if __name__ == "__main__":
    prompt = "Hey, what is your name?"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token = "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")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token = "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")
    result, tokenized, attn_m = get_completion(prompt, tokenizer, model)
    print(result)
    print(attn_m.shape)
    sparse = attn_m.to_sparse()
    attention_view(sparse, filename= "Llama2-7b-chat-hf.json")
