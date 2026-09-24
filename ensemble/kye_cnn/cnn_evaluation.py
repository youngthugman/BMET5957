"""Full-coverage inference helpers from Kye-1D_CNN."""
import numpy as np

from .cnn_cache import CONTEXT_WINDOWS


class ContextDataset:
    def __init__(self, features, labels=None, indices=None):
        import torch
        self.torch = torch; self.features = [np.asarray(x, np.float32) for x in features]
        self.labels = labels
        mapping = [(patient, second) for patient, value in enumerate(features) for second in range(len(value))]
        self.mapping = mapping if indices is None else [mapping[index] for index in indices]

    def __len__(self): return len(self.mapping)

    def __getitem__(self, index):
        patient, second = self.mapping[index]; values = self.features[patient]
        contexts = []
        for window in CONTEXT_WINDOWS:
            radius = window // 2
            positions = np.clip(np.arange(second - radius, second + radius + 1), 0, len(values) - 1)
            contexts.append(self.torch.from_numpy(values[positions]))
        if self.labels is None: return contexts
        return contexts, self.torch.tensor(float(self.labels[patient][second]))


def predict_all(model, features, device, batch_size):
    import torch
    dataset = ContextDataset(features)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    result = []; model.eval()
    with torch.no_grad():
        for contexts in loader:
            result.append(torch.sigmoid(model([value.to(device) for value in contexts])).cpu().numpy())
    probability = np.concatenate(result)
    expected = sum(map(len, features))
    assert probability.size == expected, "CNN validation rows != expected validation annotation rows"
    return probability
