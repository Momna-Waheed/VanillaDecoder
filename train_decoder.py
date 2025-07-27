import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import os
import json
from decoder_lm import DecoderLM
from char_tokenizer import CharTokenizer
import csv

# ---- Config ----
TEXT_PATH = "./Decoder/cleaned_urdu_news.txt"
VOCAB_PATH = "./Decoder/vocab.json"
LOG_PATH = "loss_log.csv"           
BATCH_SIZE = 32
MAX_LEN = 128
EPOCHS = 40
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---- Dataset ----
class UrduDataset(Dataset):
    def __init__(self, lines, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = [line.strip() for line in lines if len(line.strip()) > 0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx][:self.max_len - 2]
        encoded = self.tokenizer.encode(text)
        return torch.tensor(encoded, dtype=torch.long)

def collate_fn(batch):
    batch = [item for item in batch if len(item) > 0]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    return inputs, targets

# ---- Load Data ----
with open(TEXT_PATH, encoding="utf-8") as f:
    lines = f.readlines()
random.shuffle(lines)
split = int(0.9 * len(lines))
train_lines, val_lines = lines[:split], lines[split:]

# ---- Tokenizer & Dataloaders ----
tokenizer = CharTokenizer(TEXT_PATH)
tokenizer.load_vocab(VOCAB_PATH)

train_dataset = UrduDataset(train_lines, tokenizer, MAX_LEN)
val_dataset = UrduDataset(val_lines, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ---- Model, Loss, Optimizer ----
model = DecoderLM(vocab_size=len(tokenizer.char2idx)).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.AdamW(model.parameters(), lr=LR)

ckpt_path = os.path.join(CHECKPOINT_DIR, "decoder_epoch29.pt")
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
print(f"✅ Resumed training from {ckpt_path}")



# ---- Training Loop ----
for epoch in range(30, EPOCHS + 1):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)

    # Logging
    print(f"\n📘 Epoch {epoch}/{EPOCHS}")
    print(f"   ✅ Train Loss: {avg_train_loss:.4f}")
    print(f"   🧪 Val Loss:   {avg_val_loss:.4f}")

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{avg_train_loss:.4f}", f"{avg_val_loss:.4f}"])


    # Save checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"decoder_epoch{epoch}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"   💾 Saved checkpoint: {ckpt_path}")
