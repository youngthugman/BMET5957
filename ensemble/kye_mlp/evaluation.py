"""Kye-MLP full-coverage inference helper."""
import numpy as np


def predict_all(model, features, device, batch_size):
    import torch
    model.eval(); result = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            value = torch.from_numpy(features[start:start + batch_size]).to(device)
            result.append(torch.sigmoid(model(value)).cpu().numpy())
    probability = np.concatenate(result)
    assert probability.size == len(features), "MLP validation rows != expected annotation rows"
    return probability
