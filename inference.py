import torch
import torch.nn.functional as F
from decoder_lm import DecoderLM
from char_tokenizer import CharTokenizer

# ---- Config ----
CHECKPOINT = "checkpoints/decoder_epoch25.pt"
VOCAB_PATH = "./Decoder/vocab.json"
TEXT_PATH = "./Decoder/cleaned_urdu_news.txt"
BEAM_WIDTH = 5
MAX_LEN = 128
REPETITION_PENALTY = 1.0  # Moderate penalty to reduce looping
TOP_K = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Load tokenizer ----
tokenizer = CharTokenizer(TEXT_PATH)
tokenizer.load_vocab(VOCAB_PATH)

# ---- Load model ----
model = DecoderLM(vocab_size=len(tokenizer.char2idx)).to(DEVICE)
state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
if 'model_state_dict' in state_dict:
    model.load_state_dict(state_dict['model_state_dict'])
else:
    model.load_state_dict(state_dict)
model.eval()

def beam_search_with_prefix(
    model,
    tokenizer,
    prefix="",
    beam_width=5,
    max_len=128,
    top_k_sampling=False,
    top_k=10,
    repetition_penalty=1.0,
    return_all=False
):
    device = next(model.parameters()).device
    eos = tokenizer.eos_token_id

    prefix_ids = tokenizer.encode(prefix)[:-1]
    beams = [(prefix_ids, 0.0)]  # (sequence, score)

    for _ in range(max_len - len(prefix_ids)):
        all_candidates = []
        for seq, score in beams:
            if seq[-1] == eos:
                all_candidates.append((seq, score))
                continue

            input_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)

            probs = F.log_softmax(logits[0, -1], dim=-1)

            # Apply repetition penalty
            for token_id in set(seq):
                probs[token_id] /= repetition_penalty

            if top_k_sampling:
                topk = torch.topk(probs, k=top_k)
                probs_topk = F.softmax(topk.values, dim=-1)
                for _ in range(beam_width):
                    sampled_idx = torch.multinomial(probs_topk, 1).item()
                    next_token = topk.indices[sampled_idx].item()
                    new_seq = seq + [next_token]
                    new_score = score + topk.values[sampled_idx].item()
                    all_candidates.append((new_seq, new_score))
            else:
                topk = torch.topk(probs, k=beam_width)
                for idx, log_prob in zip(topk.indices.tolist(), topk.values.tolist()):
                    new_seq = seq + [idx]
                    new_score = score + log_prob
                    all_candidates.append((new_seq, new_score))

        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[-1] == eos for seq, _ in beams):
            break

    if return_all:
        return [tokenizer.decode(seq) for seq, _ in beams]
    return tokenizer.decode(beams[0][0])

# ---- Run Inference ----
custom_prefix = "پاکستان"  # ⬅️ Change this to try other prefixes

results = beam_search_with_prefix(
    model,
    tokenizer,
    prefix=custom_prefix,
    beam_width=BEAM_WIDTH,
    max_len=MAX_LEN,
    top_k_sampling=False,
    top_k=TOP_K,
    repetition_penalty=REPETITION_PENALTY,
    return_all=True
)

print(f"\n📝 Prefix: '{custom_prefix}'")
print("📜 Beam Search Outputs:")
for i, r in enumerate(results):
    print(f"{i+1}. {r}")
