import torch
from torch.utils.data import Dataset


class NDataSet(Dataset):
    def __init__(self, input_data: torch.Tensor, output_data: torch.Tensor, reshape=None, transforms=None):
        """
        :param input_data: 数据
        :param output_data: 标签
        :param reshape: 对数据的变形
        :param transforms: 对数据做变换
        """
        self.data = input_data
        self.transforms = None
        if reshape:
            self.data = self.data.reshape(reshape)
        if transforms:
            self.transforms = transforms
        self.targets = output_data

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx: int):
        data = self.data[idx, :]
        target = self.targets[idx]

        if self.transforms:
            data = self.transforms(data)

        return data, target
