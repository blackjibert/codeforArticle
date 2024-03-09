import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def split_dataset_indices(indices_per_class, num_servers, server_id):
    """
    将每个类别的数据集索引分成 num_servers 份，返回第 server_id 份的索引
    """
    num_classes = len(indices_per_class)
    indices_per_server = [[] for _ in range(num_servers)]
    for c in range(num_classes):
        indices = indices_per_class[c]
        num_indices = len(indices)
        indices_per_server[c % num_servers].extend(indices[server_id:num_indices:num_servers])
    return indices_per_server

# 定义数据集变换
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# 下载并加载数据集
train_dataset = datasets.CIFAR10(root='./datac', train=True, download=True, transform=transform)

# 将数据集索引分成4份，假设当前是第0个服务器
indices_per_class = [[] for _ in range(10)]
for idx, value in enumerate(train_dataset.targets):
    indices_per_class[value].append(idx)
indices_per_server = split_dataset_indices(indices_per_class, 4, 0)

# 根据服务器索引获取对应的数据集
is_kept_mask = torch.tensor([x in indices_per_server for x in range(len(train_dataset.targets))])
data = torch.masked_select(torch.tensor(train_dataset.data.transpose(0, 3)), is_kept_mask).view(-1, 32, 32, 3).transpose(0, 3, 1, 2)
labels = torch.tensor([train_dataset.targets[i] for i in range(len(train_dataset.targets)) if is_kept_mask[i]])

# 打印数据和标签的形状
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)
