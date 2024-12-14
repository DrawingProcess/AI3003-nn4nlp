import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import time
import random
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# Ensure the 'results' directory exists
if not os.path.exists('results'):
    os.makedirs('results')

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])

# Read in the data
train_data = list(read_dataset("../data/classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)  # Now UNK is used for unknown words

# Create index-to-word mapping
i2w = {index: word for word, index in w2i.items()}
i2t = {index: tag for tag, index in t2i.items()}

dev_data = list(read_dataset("../data/classes/test.txt"))
nwords = len(w2i)
ntags = len(t2i)

# Add PAD_IDX
PAD_IDX = len(w2i)
w2i['<pad>'] = PAD_IDX
i2w[PAD_IDX] = '<pad>'
nwords = len(w2i)

# Custom dataset class to work with DataLoader
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Collate function for padding sentences in each batch
def collate_fn(batch):
    sentences, tags = zip(*batch)
    sentences = [torch.tensor(sent) for sent in sentences]
    tags = torch.tensor(tags)
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=PAD_IDX)
    return padded_sentences, tags

# Define DataLoader
BATCH_SIZE = 32
train_loader = DataLoader(TextDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(TextDataset(dev_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Define the model
EMB_SIZE = 256
HID_SIZE = 256

# Attention 모듈 정의
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)  # Bi-directional이므로 hidden_size * 2

    def forward(self, rnn_out, mask):
        # rnn_out: [batch_size x seq_len x hidden_size*2]
        # mask: [batch_size x seq_len]

        # Attention scores 계산
        attn_scores = self.attention(rnn_out).squeeze(2)  # [batch_size x seq_len]

        # 패딩 위치에 대한 마스크 적용
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        # Attention weights 계산
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch_size x seq_len]

        # Context vector 계산
        attn_weights_unsqueezed = attn_weights.unsqueeze(1)  # [batch_size x 1 x seq_len]
        context = torch.bmm(attn_weights_unsqueezed, rnn_out).squeeze(1)  # [batch_size x hidden_size*2]

        return context, attn_weights

# RNNModel 수정
class RNNModel(nn.Module):
    def __init__(self, nwords, ntags, emb_size, hidden_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(nwords, emb_size, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size)  # Attention 모듈 사용
        self.fc = nn.Linear(hidden_size * 2, ntags)
        self.dropout = nn.Dropout(0.5)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'embedding' not in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, sentences):
        embeds = self.dropout(self.embedding(sentences))
        rnn_out, _ = self.rnn(embeds)

        # 패딩 위치 마스크 생성
        mask = (sentences != PAD_IDX)

        # Attention 모듈을 통해 context vector와 attention weights 계산
        context, attn_weights = self.attention(rnn_out, mask)

        # 출력 레이어
        logits = self.fc(context)
        return logits, attn_weights



# Visualization function for attention weights
def visualize_attention(sentence, attention_weights, idx, mode='train'):
    
    print(f"Sentence shape: {sentence.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    # Convert indices back to words, excluding padding
    words = [i2w[token.item()] for token in sentence if token.item() != PAD_IDX]

    # Convert attention_weights to a 1D numpy array
    weights = attention_weights.cpu().detach().numpy()  # Shape: [seq_len]

    # Create a mask for non-padding tokens
    mask = (sentence != PAD_IDX).cpu().numpy()  # Shape: [seq_len]

    # Apply mask to attention weights
    weights = weights[mask]

    # Ensure weights and words have the same length
    if len(weights) != len(words):
        print(f"Length of weights ({len(weights)}) does not match length of words ({len(words)}). Adjusting weights.")
        # Adjust to the minimum length
        min_len = min(len(weights), len(words))
        weights = weights[:min_len]
        words = words[:min_len]

    # Normalize weights (for visualization)
    weights = weights / weights.sum()

    # Use matplotlib to visualize attention weights
    plt.figure(figsize=(10, 2))
    plt.bar(range(len(words)), weights, tick_label=words, align='center')
    plt.xticks(rotation=45)
    plt.xlabel('Words')
    plt.ylabel('Attention Weight')
    plt.title(f'Attention Weights - Example {idx} ({mode})')
    plt.tight_layout()
    plt.savefig(f'results/attention_{mode}_{idx}.png')
    plt.close()





model = RNNModel(nwords, ntags, EMB_SIZE, HID_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training started: sentiment-rnn-minibatch.py")
# Training loop
max_test_accuracy = 0.0
for ITER in range(100):  # Reduced iterations for brevity
    # Perform training
    train_loss = 0.0
    start = time.time()
    model.train()

    for batch_idx, (sentences, tags) in enumerate(tqdm(train_loader)):
        logits, _ = model(sentences)
        loss = criterion(logits, tags)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Visualize attention for the first batch of each epoch
        if batch_idx == 0 and ITER % 1 == 0:
            with torch.no_grad():
                logits, attn_weights = model(sentences)
                # Visualize attention for the first example in the batch
                visualize_attention(sentences[0], attn_weights[0], idx=ITER, mode='train')
    
    print(f"iter {ITER}: train loss/sent={train_loss / len(train_loader):.4f}, time={time.time() - start:.2f}s")

    # Perform evaluation
    model.eval()
    test_correct = 0.0
    with torch.no_grad():
        for batch_idx, (sentences, tags) in enumerate(tqdm(dev_loader)):
            logits, attn_weights = model(sentences)
            predict = torch.argmax(logits, dim=1)
            test_correct += (predict == tags).sum().item()

            # Visualize attention for the first batch in evaluation
            if batch_idx == 0 and ITER % 1 == 0:
                # Visualize attention for the first example in the batch
                visualize_attention(sentences[0], attn_weights[0], idx=ITER, mode='dev')

    test_accuracy = test_correct / len(dev_data)
    if max_test_accuracy < test_accuracy:
        max_test_accuracy = test_accuracy
    print(f"iter {ITER}: test acc={test_accuracy:.4f}")
    
print("max test acc=%.4f" % (max_test_accuracy))
