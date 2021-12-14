# coding: utf-8

import pandas as pd
import torch
import numpy as np
import random
import kss
import re
from konlpy.tag import Komoran
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


# ## 1. 가사 데이터

lyrics = pd.read_csv("data/lyrics.csv").dropna()

alphabet = re.compile("[a-zA-Z]")
lyrics = lyrics.assign(
    alphabet_ratio = pd.Series([len(re.findall("[a-zA-Z]", lyric)) for lyric in lyrics.lyrics]) / lyrics.lyrics.apply(len),
).query("alphabet_ratio <= 0.25").drop("alphabet_ratio", axis=1)
lyrics["lyrics"] = lyrics["lyrics"].str.replace("\r\n", ". ")
lyrics = lyrics["lyrics"].tolist()
corpus = pd.Series(corpus).drop_duplicates().reset_index(drop=True)
length = corpus.str.replace("[\s.]", "", regex=True).apply(len)
alphabet_length =  pd.Series([len(re.findall("[a-zA-Z]", lyric)) for lyric in corpus])
corpus = corpus[alphabet_length.div(length) < 0.25]
duplicated = corpus.str.replace("[\s.]", "", regex=True).duplicated(keep="last")
corpus = corpus[~duplicated]
corpus = corpus.reset_index(drop=True)
corpus = corpus.str.replace(".", "")

with open("lyrics.txt", "w", encoding="utf-8") as f:
    for line in corpus.values:
        f.write(line.strip() + "\n")


## 2. 문장 데이터
texts = pd.read_csv("data/text.tsv", sep=r"\t", encoding="utf-8").dropna()
with open("data/text.txt", "w") as f:
    for sentence in texts.sentence2.values:
        f.write(sentence.strip() + "\n")


## 3. 전처리
with open("data/text.txt", "r") as f:
    query = [l.strip() for l in f.readlines()]
with open("data/lyrics.txt", "r") as f:
    corpus = [l.strip() for l in f.readlines()]


## 4. 유사도 매칭

embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
query = query.tolist()
corpus = corpus.tolist()
corpus_embeddings = embedder.encode(corpus, batch_size=128, convert_to_tensor=True)
query_embeddings = embedder.encode(query, batch_size=128, convert_to_tensor=True)
match = []
for i, query_embedding in tqdm(enumerate(query_embeddings)):
    cos_scores = util.pytorch_cos_sim(query_embeddings[i], corpus_embeddings).flatten().cpu().numpy()
    idxmax = cos_scores.argmax()
    result = dict(query = query[i], corpus_idx=corpus[idxmax], cos_sim=cos_scores[idxmax])
    match.append(result)
match = pd.DataFrame(match)
match.to_csv("match.csv", index=False, encoding="utf-8")
