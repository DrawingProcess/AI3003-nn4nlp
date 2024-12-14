import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu


# 시드 값을 설정합니다.
SEED = 42

# Python의 기본 랜덤 모듈 시드 고정
random.seed(SEED)

# Numpy 시드 고정
np.random.seed(SEED)

# PyTorch 시드 고정
torch.manual_seed(SEED)

# CUDA 사용 시에도 시드를 고정합니다.
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # 멀티 GPU 사용 시

# CuDNN 관련 설정을 고정하여 비결정적인 연산을 방지합니다.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data paths
train_src_file = "../data/parallel/train.ja"
train_trg_file = "../data/parallel/train.en"
dev_src_file = "../data/parallel/dev.ja"
dev_trg_file = "../data/parallel/dev.en"
test_src_file = "../data/parallel/test.ja"
test_trg_file = "../data/parallel/test.en"

w2i_src = defaultdict(lambda: len(w2i_src))
w2i_trg = defaultdict(lambda: len(w2i_trg))

def read(fname_src, fname_trg):
    with open(fname_src, "r", encoding='utf-8') as f_src, open(fname_trg, "r", encoding='utf-8') as f_trg:
        for line_src, line_trg in zip(f_src, f_trg):
            sent_src = [w2i_src[x] for x in line_src.strip().split()] + [w2i_src['</s>']]
            sent_trg = [w2i_trg['<s>']] + [w2i_trg[x] for x in line_trg.strip().split()] + [w2i_trg['</s>']]
            yield (sent_src, sent_trg)

# Build vocab and load data
train_data = list(read(train_src_file, train_trg_file))
dev_data = list(read(dev_src_file, dev_trg_file))
test_data = list(read(test_src_file, test_trg_file))

# Special tokens
unk_src = w2i_src['<unk>']
eos_src = w2i_src['</s>']
pad_src = w2i_src['<pad>']
w2i_src = defaultdict(lambda: unk_src, w2i_src)
unk_trg = w2i_trg['<unk>']
eos_trg = w2i_trg['</s>']
sos_trg = w2i_trg['<s>']
pad_trg = w2i_trg['<pad>']
w2i_trg = defaultdict(lambda: unk_trg, w2i_trg)
i2w_trg = {v: k for k, v in w2i_trg.items()}

# Dataset class
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, trg = self.data[idx]
        return torch.tensor(src), torch.tensor(trg)

# Collate function for DataLoader
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=pad_src, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=pad_trg, batch_first=True)
    return src_batch, trg_batch

# Create DataLoader
BATCH_SIZE = 32  # 배치 사이즈 증가
train_loader = DataLoader(TranslationDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(TranslationDataset(dev_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Attention Mechanism: Decoder-Encoder Concat Attention (Luong's Global Attention)
# Luong et al., 2015, "Effective Approaches to Attention-based Neural Machine Translation"
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        nn.init.uniform_(self.v, -0.1, 0.1)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: (batch_size, hidden_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch_size, seq_len, hidden_size)
        energy = energy.permute(0, 2, 1)  # (batch_size, hidden_size, seq_len)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # (batch_size, 1, hidden_size)
        attention = torch.bmm(v, energy).squeeze(1)  # (batch_size, seq_len)
        attention = attention.masked_fill(mask == 0, -1e10)  # 마스크 적용
        return torch.softmax(attention, dim=1)

# Seq2Seq Model with Attention
class Seq2SeqWithAttention(nn.Module):
    def __init__(self, nwords_src, nwords_trg, embed_size, hidden_size, dropout_p=0.3):
        super(Seq2SeqWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_src = nn.Embedding(nwords_src, embed_size)
        self.embedding_trg = nn.Embedding(nwords_trg, embed_size)
        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.decoder_lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, nwords_trg)
        self.dropout = nn.Dropout(dropout_p)


    def create_mask(self, src):
        # src: (batch_size, src_len)
        mask = (src != pad_src)
        return mask

    def forward(self, src, trg):
        src_mask = self.create_mask(src)
        embedded_src = self.dropout(self.embedding_src(src))
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)
        # Sum bidirectional outputs
        encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]

        # Prepare initial hidden and cell states for decoder
        hidden = torch.tanh(hidden[0] + hidden[1]).unsqueeze(0)  # (1, batch_size, hidden_size)
        cell = torch.tanh(cell[0] + cell[1]).unsqueeze(0)        # (1, batch_size, hidden_size)

        embedded_trg = self.dropout(self.embedding_trg(trg))
        outputs = []
        for i in range(trg.size(1)):
            trg_t = embedded_trg[:, i].unsqueeze(1)  # (batch_size, 1, embed_size)
            attention_weights = self.attention(hidden[-1], encoder_outputs, src_mask)
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_size)
            lstm_input = torch.cat([trg_t, context], dim=2)
            output, (hidden, cell) = self.decoder_lstm(lstm_input, (hidden, cell))
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)  # (batch_size, trg_len, hidden_size)
        logits = self.fc(outputs)
        return logits

    def generate(self, src, sos_trg, eos_trg, max_len, beam_width=3):
        src_mask = self.create_mask(src)
        embedded_src = self.embedding_src(src)
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)
        encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]

        hidden = torch.tanh(hidden[0] + hidden[1]).unsqueeze(0)
        cell = torch.tanh(cell[0] + cell[1]).unsqueeze(0)

        # Beam Search 초기화
        sequences = [[list(), 1.0, hidden, cell]]
        for _ in range(max_len):
            all_candidates = []
            for seq, score, hidden, cell in sequences:
                if len(seq) > 0 and seq[-1] == eos_trg:
                    all_candidates.append((seq, score, hidden, cell))
                    continue
                trg_tensor = torch.LongTensor([seq[-1] if len(seq) > 0 else sos_trg]).unsqueeze(0).to(src.device)
                trg_embedded = self.embedding_trg(trg_tensor)
                attention_weights = self.attention(hidden[-1], encoder_outputs, src_mask)
                context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
                lstm_input = torch.cat([trg_embedded, context], dim=2)
                output, (hidden, cell) = self.decoder_lstm(lstm_input, (hidden, cell))
                output = self.fc(output.squeeze(1))
                probs = torch.softmax(output, dim=1)
                topk = probs.topk(beam_width)
                for i in range(beam_width):
                    word_idx = topk.indices[0][i].item()
                    word_prob = topk.values[0][i].item()
                    candidate = (seq + [word_idx], score * -math.log(word_prob), hidden, cell)
                    all_candidates.append(candidate)
            # 선택된 후보들 중 상위 beam_width 개를 선택
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            sequences = ordered[:beam_width]

        # 가장 높은 확률의 시퀀스를 선택
        best_seq = sequences[0][0]
        trg_sentence = [i2w_trg[idx] for idx in best_seq if idx != eos_trg]
        return trg_sentence

# Model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_SIZE = 4096  # 임베딩 사이즈 증가
HIDDEN_SIZE = 1024  # 히든 사이즈 증가
model = Seq2SeqWithAttention(len(w2i_src), len(w2i_trg), EMBED_SIZE, HIDDEN_SIZE, dropout_p=0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)  # 학습률 조절
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)

# Ignore padding token in loss calculation
criterion = nn.CrossEntropyLoss(ignore_index=pad_trg)

# Training and evaluation functions
def train_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    total_words = 0

    for src_batch, trg_batch in tqdm(train_loader):
        src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
        trg_input = trg_batch[:, :-1]
        trg_output = trg_batch[:, 1:]

        outputs = model(src_batch, trg_input)
        outputs = outputs.view(-1, outputs.shape[-1])
        trg_output = trg_output.contiguous().view(-1)

        # Only count non-padding tokens
        non_pad_elements = trg_output.ne(pad_trg).sum().item()

        loss = criterion(outputs, trg_output)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # Gradient Clipping
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * non_pad_elements  # multiply by non-padded tokens
        total_words += non_pad_elements

    return total_loss / total_words

def evaluate(model, dev_loader):
    model.eval()
    total_loss = 0
    total_words = 0

    with torch.no_grad():
        for src_batch, trg_batch in tqdm(dev_loader):
            src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
            trg_input = trg_batch[:, :-1]
            trg_output = trg_batch[:, 1:]

            outputs = model(src_batch, trg_input)
            outputs = outputs.view(-1, outputs.shape[-1])
            trg_output = trg_output.contiguous().view(-1)

            # Only count non-padding tokens
            non_pad_elements = trg_output.ne(pad_trg).sum().item()

            loss = criterion(outputs, trg_output)

            total_loss += loss.item() * non_pad_elements  # multiply by non-padded tokens
            total_words += non_pad_elements

    return total_loss / total_words

print("Training started: enc_dec-attention.py")
# Training loop
max_test_bleu = 0.0
for epoch in range(300):  # 에폭 수 조절
    train_loss = train_epoch(model, train_loader, optimizer)
    dev_loss = evaluate(model, dev_loader)
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}, Perplexity: {math.exp(dev_loss):.4f}")

    # Generate translation
    test_loader = DataLoader(TranslationDataset(test_data), batch_size=1, shuffle=False, collate_fn=collate_fn)
    bleu_scores = []
    for src_batch, trg_batch in test_loader:
        src_batch = src_batch.to(device)
        translated_sent = model.generate(src_batch, sos_trg, eos_trg, max_len=50, beam_width=5)
        translated_sent_str = ' '.join(translated_sent)

        # Convert target sentence to words using i2w_trg
        reference_sent = [i2w_trg[token.item()] for token in trg_batch[0] if token.item() != pad_trg]
        reference_sent = reference_sent[1:-1]  # Remove <s> and </s>

        # Calculate BLEU score
        bleu_score = sentence_bleu([reference_sent], translated_sent, weights=(0.5, 0.5, 0, 0))
        bleu_scores.append(bleu_score)

    # Average BLEU score across the test set
    test_bleu = sum(bleu_scores) / len(bleu_scores)

    if max_test_bleu < test_bleu:
        max_test_bleu = test_bleu
    print(f"epoch {epoch}: test acc={test_bleu:.4f}")

print("max test bleu=%.4f" % (max_test_bleu))
