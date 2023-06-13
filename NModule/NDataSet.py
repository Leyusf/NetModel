import torch
from torch.utils.data import Dataset

from NModule.NCV import read_dataset


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


class ObjectDetectDataset(torch.utils.data.Dataset):
    """一个用于目标检测数据集的自定义数据集"""

    def __init__(self, data_dir, is_train=True, transforms=None):
        super().__init__()
        self.images, self.labels = read_dataset(data_dir, is_train)
        self.transforms = transforms
        print('read ' + str(len(self.images)) + (f' training examples' if
                                                   is_train else f' validation examples'))

    def __getitem__(self, idx):
        return self.images[idx].float(), self.labels[idx]

    def __len__(self):
        return len(self.images)
