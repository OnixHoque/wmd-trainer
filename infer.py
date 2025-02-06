import torch
import pickle
import argparse
import numpy as np
from torchkge.models import TransE
from torchkge.utils import DataLoader

# Load dataset mappings
def load_mappings(filepath):
    with open(filepath, 'rb') as f:
        ent2ix, rel2ix = pickle.load(f)
    return ent2ix, rel2ix

# Load model
def load_model(model_path, ent_size, rel_size, emb_dim=100):
    model = TransE(ent_size, rel_size, emb_dim, norm=1, dissimilarity_type='L2')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Perform inference
def infer(model, ent2ix, rel2ix, head, tail, rel, top_n):
    ix2ent = {v: k for k, v in ent2ix.items()}
    ix2rel = {v: k for k, v in rel2ix.items()}
    candidates = []
    
    if head == -1:
        for h in range(len(ent2ix)):
            score = model.scoring_function(torch.tensor([h]), torch.tensor([rel]), torch.tensor([tail])).item()
            candidates.append((h, rel, tail, score))
    elif tail == -1:
        for t in range(len(ent2ix)):
            score = model.scoring_function(torch.tensor([head]), torch.tensor([rel]), torch.tensor([t])).item()
            candidates.append((head, rel, t, score))
    elif rel == -1:
        for r in range(len(rel2ix)):
            score = model.scoring_function(torch.tensor([head]), torch.tensor([r]), torch.tensor([tail])).item()
            candidates.append((head, r, tail, score))
    
    # Sort by score (lower is better for TransE with L2 distance)
    candidates.sort(key=lambda x: x[3])
    return [(ix2ent[h], ix2rel[r], ix2ent[t], s) for h, r, t, s in candidates[:top_n]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model params file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset mappings file")
    parser.add_argument("--num", type=int, required=True, help="Number of top predictions to return")
    parser.add_argument("--head", type=int, required=True, help="Head entity index (-1 if unknown)")
    parser.add_argument("--rel", type=int, required=True, help="Relation index (-1 if unknown)")
    parser.add_argument("--tail", type=int, required=True, help="Tail entity index (-1 if unknown)")
    parser.add_argument("--output", type=str, required=True, help="Output TSV file")
    args = parser.parse_args()
    
    ent2ix, rel2ix = load_mappings(args.dataset)
    model = load_model(args.model, len(ent2ix), len(rel2ix))
    results = infer(model, ent2ix, rel2ix, args.head, args.tail, args.rel, args.num)
    
    # Save results
    with open(args.output, 'w') as f:
        f.write("Head\tRelation\tTail\tScore\n")
        for h, r, t, s in results:
            f.write(f"{h}\t{r}\t{t}\t{s}\n")
    
    print(f"Inference complete. Results saved to {args.output}")


# python infer.py --model model_params.pt --dataset dataset_params.pt --num 5 --head -1 --rel 2 --tail 10 --output results.tsv
