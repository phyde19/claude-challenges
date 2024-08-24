from torch.utils.data import Dataset, ConcatDataset, Subset, random_split
from torchvision import datasets, transforms
import math

def random_sample(dataset: Dataset, n: int):
  sample, _ = random_split(dataset, [n, len(dataset) - n])
  return sample

class MNISTAnomalyDetection:
  def __init__(self, test_ratio: float = .2,  test_anomaly_ratio: float = 0.1, normal_labels: list[int] = [0]):
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    combined = ConcatDataset([mnist_train, mnist_test])
    normal = Subset(combined, [i for i, (_, label) in enumerate(combined) if label in normal_labels])
    anomaly = Subset(combined, [i for i, (_, label) in enumerate(combined) if label not in normal_labels])
    train_dataset, test_dataset = self._test_train_split(normal, anomaly, test_ratio, test_anomaly_ratio)
    self._test_dataset = test_dataset
    self._train_dataset = train_dataset

  def _test_train_split(self, normal: Dataset, anomaly: Dataset, test_ratio: float, test_anomaly_ratio: float):
    train_ratio = 1 - test_ratio
    test_normal_ratio = 1 - test_anomaly_ratio
    abs_test_normal_ratio = test_ratio * test_normal_ratio
    num_train_normal = math.floor(len(normal) * train_ratio / (train_ratio + abs_test_normal_ratio))
    num_test_normal = len(normal) - num_train_normal
    num_test_anomaly = math.floor(num_test_normal * (test_anomaly_ratio / test_normal_ratio))
    train_normal, test_normal = random_split(normal, [num_train_normal, num_test_normal])
    test_anomaly = random_sample(anomaly, num_test_anomaly)
    return train_normal, ConcatDataset([test_normal, test_anomaly])

  def train_dataset(self):
    return self._train_dataset
  
  def test_dataset(self):
    return self._test_dataset
