import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


class DecoderLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3, dropout=0.1, max_len=512):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory=None):
        """
        tgt: [B, T] target character IDs
        memory: not used (for future integration with encoder)
        """
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.decoder(tgt_emb, memory=torch.zeros_like(tgt_emb), tgt_mask=tgt_mask)
        logits = self.output_proj(output)
        return logits  # shape: [B, T, vocab_size]
