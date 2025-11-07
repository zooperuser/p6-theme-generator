from app import Config, MoodPaletteGenerator, COLOR_DATA
import os, numpy as np, random

cfg = Config()
# Instantiate generator (may try embeddings; we will stub if needed)
gen = MoodPaletteGenerator(cfg, COLOR_DATA)
# Monkeypatch embeddings if not available
if gen.color_embeddings is None:
    random.seed(0)
    np.random.seed(0)
    gen.color_embeddings = np.random.rand(len(COLOR_DATA), 8).astype('float32')
    gen.embedding_client.encode = lambda x: np.random.rand(1,8).astype('float32')

prompt = 'Test duplicate prompt'
print('Generating first time...')
gen.generate(prompt)
print('Generating second time (should not duplicate)...')
gen.generate(prompt)

# Read resulting log file
with open(cfg.LOG_FILE, 'r', encoding='utf-8') as f:
    lines = [l.rstrip('\n') for l in f]

print('\n--- Log File Content ---')
print('\n'.join(lines))
count = sum(1 for l in lines if l.strip() == f'- {prompt}')
print(f"Occurrences of '{prompt}':", count)
assert count == 1, 'Prompt was duplicated in log!'
print('Deduplication test passed.')
