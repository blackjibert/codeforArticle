#备用
import sys
import syft as sy
from multiprocessing import Process, Pool, get_context

from torchvision import datasets, transforms
from split_dataset import split_dataset_indices

from custom_server import CustomWebsocketServerWorker
 
def main(n_server, stdev):
    print("Splitting dataset...")
    targetss = datasets.CIFAR10(
        root="./datacifar10",
        train=True,
        download=True
    ).targets
    indices_per_class = [[] for _ in range(10)]
    for idx, value in enumerate(targetss):
        indices_per_class[value].append(idx)
    indices_per_server = split_dataset_indices(indices_per_class, n_server, stdev)

    print("Starting servers...")
    with get_context("spawn").Pool(processes=n_server) as pool:
        pool.starmap(run_server, zip(range(n_server), indices_per_server))


def run_server(i, indices):
    import torch  # Each process should import torch to allow parallelization?

    hook = sy.TorchHook(torch)
    server = CustomWebsocketServerWorker(
        id=f"dataserver-{i}",
        host="0.0.0.0",
        # host="127.0.0.1",  # linux下为0.0.0.0, window下为127.0.0.1;不太确定
        port=f"{8777 + i}",
        hook=hook
    )

    cifar110 = datasets.CIFAR10(
        root="./datacifar10",
        train=True,
        download=True,
        transform=transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomCrop(32, padding=4),
            # transforms.ToTensor(),
            # # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Resize((227, 227)),
            # transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            # transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
            #                      [0.24703223, 0.24348513, 0.26158784])
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),

    )
    tensor_targets = torch.tensor(cifar110.targets)
    tensor_data = torch.from_numpy(cifar110.data)
    is_kept_mask = torch.tensor([x in indices for x in range(len(cifar110.targets))])
    dataset = sy.BaseDataset(
        data=torch.masked_select(cifar110.data.transpose(0, 3), is_kept_mask).view(3, 32, 32, -1).transpose(3, 0),
        # data=torch.masked_select(tensor1.data.transpose(0, 2), is_kept_mask).view(28, 28, -1).transpose(2, 0),
        targets=torch.masked_select(cifar110.targets, is_kept_mask),
        transform=cifar110.transform
    )

    server.add_dataset(dataset, key="cifar10")
    print(f"Server {i} started")
    server.start()


if __name__ == "__main__":
    # Argument parsing
    try:
        # N_SERVER = 4  # 数据服务器数量
        N_SERVER = int(sys.argv[1])
        STDEV = int(sys.argv[2])
        # STDEV = 1000  # 数据不平衡程度
        print(f"Will start {N_SERVER} servers (data owners) with stdev of {STDEV}")
    except Exception as e:
        print(e)
        sys.exit()

    main(N_SERVER, STDEV)
