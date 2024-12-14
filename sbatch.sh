#!/usr/bin/bash

#SBATCH -J nlp-05-attention-attention_GradientClipping_ReduceLROnPlateau
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.outs

cat $0
pwd
which python
hostname

# cd 01-intro
# python bow.py
# python cbow.py
# python deep_cbow.py

# cd 02-lm
# python nn-lm-batch.py
# python nn-lm-nobatch.py

# cd 03-rnn
# python sentiment-rnn.py
# python sentiment-rnn-minibatch.py
# python sentiment-rnn-attention-minibatch.py

# cd 04-condlm
# python enc_dec.py
# python enc_dec-gru.py
# python enc_dec-attention.py


cd 05-attention
# python attention.py
python attention_modify.py
# python attention_self.py
# python attention_multihead.py
# python attention_scaleddotproduct.py
# python attention_positionalencoding.py

exit 0