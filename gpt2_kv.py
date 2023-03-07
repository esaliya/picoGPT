# Copied from https://raw.githubusercontent.com/immortal3/picoGPT/d663909cfb026f15cef1ed64a4e929506beae10f/gpt2.py

import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    # normalize x to have mean=0 and var=1 over last axis
    x = (x - mean) / np.sqrt(variance + eps)
    return g * x + b  # scale and offset with gamma/beta params


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x


# [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
def attention(q, k, v, mask):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


# [n_seq, n_embd] -> [n_seq, n_embd]
def mha(x, c_attn, c_proj, n_head, kvcache=None):
    # qkv projection
    # when we pass kvcache, n_seq = 1. so we will compute new_q, new_k and new_v
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    if kvcache:
        # qkv
        new_q, new_k, new_v = qkv  # new_q, new_k, new_v = [1, n_embd]
        old_k, old_v = kvcache
        # k = [n_seq, n_embd], where n_seq = prev_n_seq + 1
        k = np.vstack([old_k, new_k])
        # v = [n_seq, n_embd], where n_seq = prev_n_seq + 1
        v = np.vstack([old_v, new_v])
        qkv = [new_q, k, v]

    current_cache = [qkv[1], qkv[2]]

    # split into heads
    # [3, n_seq, n_embd] -> [n_head, 3, n_seq, n_embd/n_head]
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))

    # causal mask to hide future inputs from being attended to
    if kvcache:
        # when we pass kvcache, we are passing single token as input which need to attend to all previous tokens, so we create mask with all 0s
        causal_mask = np.zeros((1, k.shape[0]))
    else:
        # create triangular causal mask
        causal_mask = (1 - np.tri(x.shape[0])) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    # [n_head, 3, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]
    out_heads = [attention(q, k, v, causal_mask)
                 for q, k, v in zip(*qkv_heads)]

    # merge heads
    # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]
    x = np.hstack(out_heads)

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x, current_cache


# [n_seq, n_embd] -> [n_seq, n_embd]
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head, kvcache=None):
    # multi-head causal self attention
    attn_out, kvcache_updated = mha(layer_norm(
        x, **ln_1), **attn, n_head=n_head, kvcache=kvcache)
    x = x + attn_out  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    # [n_seq, n_embd] -> [n_seq, n_embd]
    x = x + ffn(layer_norm(x, **ln_2), **mlp)

    return x, kvcache_updated


# [n_seq] -> [n_seq, n_vocab]
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head, kvcache=None):
    if not kvcache:
        kvcache = [None]*len(blocks)
        wpe_out = wpe[range(len(inputs))]
    else:
        wpe_out = wpe[[len(inputs)-1]]
        inputs = [inputs[-1]]

    # token + positional embeddings
    x = wte[inputs] + wpe_out  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    new_kvcache = []
    for block, kvcache_block in zip(blocks, kvcache):
        # [n_seq, n_embd] -> [n_seq, n_embd]
        x, updated_cache = transformer_block(
            x, **block, n_head=n_head, kvcache=kvcache_block)
        # TODO: inplace extend new cache instead of re-saving whole
        new_kvcache.append(updated_cache)

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T, new_kvcache  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    kvcache = None
    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        # model forward pass
        logits, kvcache = gpt2(
            inputs, **params, n_head=n_head, kvcache=kvcache)
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs = np.append(inputs, [next_id])  # append prediction to input

    # only return generated ids
    return list(inputs[len(inputs) - n_tokens_to_generate:])


def main(prompt: str = "Alan Turing theorized that computers would one day become", n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(
        model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate(
        input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    import fire

    fire.Fire(main)
