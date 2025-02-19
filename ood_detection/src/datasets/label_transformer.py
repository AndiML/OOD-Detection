from torch.utils.data import Dataset as TorchDataset


class LabelTransformDataset(TorchDataset):
    """
    A dataset wrapper that applies a target transformation to each sample.
    """
    def __init__(self, base_dataset: TorchDataset, target_transform):
        self.base_dataset = base_dataset
        self.target_transform = target_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, y = self.base_dataset[index]
        return x, self.target_transform(y)
