from char_tokenizer import CharTokenizer

# Build and save tokenizer
tokenizer = CharTokenizer("./Decoder/cleaned_urdu_news.txt")
tokenizer.save_vocab("./Decoder/vocab.json")

print(tokenizer)

# Encode a sample
encoded = tokenizer.encode("پاکستان ایک خوبصورت ملک ہے۔")
print(encoded)

# Decode back
decoded = tokenizer.decode(encoded)
print(decoded)

