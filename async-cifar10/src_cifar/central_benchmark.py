import torch
from datetime import datetime
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import f1_score
from model.base_model import NetRGB
from tqdm import tqdm

mnist = torch.utils.data.DataLoader(datasets.CIFAR10(
    root="./datacifar10",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
), batch_size=10)

mnist_test = datasets.CIFAR10(
    root="../datacifar10",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)

model = NetRGB()
opt = optim.SGD(params=model.parameters(), lr=0.1)

start_time = datetime.now()
for i in range(1000):
    model.train()
    j = 0
    for data, target in tqdm(mnist):
        # 1) erase previous gradients (if they exist)
        opt.zero_grad()

        # 2) make a prediction
        pred = model(data)
        # plt.imshow(data.view(1, 1, 28, 28)[0][0])
        # plt.show()

        # 3) calculate how much we missed
        loss = F.nll_loss(input=pred, target=target)

        # print(target, pred.data, loss.item())

        # 4) figure out which weights caused us to miss
        loss.backward()

        # 5) change those weights
        opt.step()

        # 6) print our progress
        # print(loss.data)
    
        if j % 20 == 0:
            model.eval()
            y_pred = [model(datum.view(1, 3, 32, 32)).detach().numpy().argmax() for datum, _ in mnist_test] # Slow...
            f1 = f1_score(mnist_test.targets, y_pred, average='micro')
            print(f"micro-f1 score at {datetime.now() - start_time} is {f1}")
        j += 1
    

