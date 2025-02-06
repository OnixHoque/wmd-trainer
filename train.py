# pip install neo4j
import pandas as pd
from neo4j import GraphDatabase

import sys
from utils import *

uri, user, password = get_neo4j_config()
df2 = retrieve_from_neo4j(uri, user, password)
print(df2.head())

import torch
from torch import cuda
from torch.optim import Adam

from torchkge.models import TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from torchkge import KnowledgeGraph

from tqdm.autonotebook import tqdm

df2 = df2.astype(str)
kg = KnowledgeGraph(df2)
# kg_train, kg_test = kg.split_kg(share=0.8)
kg_train = kg

# Define some hyper-parameters for training

emb_dim, lr, n_epochs, b_size, margin = get_train_config()

# Define the model and criterion
model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
criterion = MarginLoss(margin)

# # Move everything to CUDA if available
if cuda.is_available():
    cuda.empty_cache()
    model.cuda()
    criterion.cuda()

# Define the torch optimizer to be used
optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
sampler = BernoulliNegativeSampler(kg_train)

if cuda.is_available():
    dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda='all')
else:
    dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda=None)

iterator = tqdm(range(n_epochs), unit='epoch')
for epoch in iterator:
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        h, t, r = batch[0], batch[1], batch[2]
        n_h, n_t = sampler.corrupt_batch(h, t, r)

        optimizer.zero_grad()

        # forward + backward + optimize
        pos, neg = model(h, t, r, n_h, n_t)
        loss = criterion(pos, neg)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    iterator.set_description(
        'Epoch {} | mean loss: {:.5f}'.format(epoch + 1,
                                              running_loss / len(dataloader)))

model.normalize_parameters()

torch.save(model.state_dict(), "model_params.pt")
import pickle
with open('dataset_params.pt', 'wb') as f:
    pickle.dump([kg.ent2ix, kg.rel2ix], 'dataset_params.pt')
