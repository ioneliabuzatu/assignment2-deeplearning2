import torch
import torchvision
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm.notebook import trange

from torchvision.models import VGG, vgg13

torch.manual_seed(1806)
torch.cuda.manual_seed(1806)

import wandb
import config

wandb.run = config.tensorboard.run


@torch.no_grad()
def evaluate(network: nn.Module, data: DataLoader, metric: callable, cuda=True) -> list:
    network.eval()

    errors = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data):

            if cuda:
                inputs = inputs.to("cuda")
                targets = targets.to("cuda")

            classification = network(inputs)
            loss_batch = metric(classification, targets)

            errors.append(loss_batch.item())

    return errors


@torch.enable_grad()
def update(network: nn.Module, data: DataLoader, loss: nn.Module, opt: optim.Optimizer, cuda=True) -> list:
    network.train()

    errors = []

    for batch_idx, (inputs, targets) in enumerate(data):

        if cuda:
            inputs = inputs.to("cuda")
            targets = targets.to("cuda")

        opt.zero_grad()

        classification = network(inputs)

        loss_batch = loss(classification, targets)

        loss_batch.backward()
        opt.step()

        errors.append(loss_batch.item())

    return errors


class CifarVGG(nn.Module):
    """ Variant of the VGG network for classifying CIFAR images. """

    def __init__(self, features: nn.Module, num_classes: int = 10):
        """
        Parameters
        ----------
        features : nn.Module
            The convolutional part of the VGG network.
        num_classes : int
            The number of output classes in the data.
        """
        super(CifarVGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def freeze_pretrained_convolution_weights(model):
    for param in model.features.parameters():
        param.requires_grad = False


vgg13_model = vgg13(pretrained=True)
vgg13_model.to("cuda")
network = CifarVGG(vgg13_model.features, num_classes=10)

freeze_pretrained_convolution_weights(network)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
cifar10_train_data = torchvision.datasets.CIFAR10("~/.pytorch", transform=transform, download=True, train=True)
cifar10_test_data = torchvision.datasets.CIFAR10("~/.pytorch", transform=transform, download=True, train=False)

train_size = 500
test_size = 500
ignored_train_data_size = len(cifar10_train_data) - 1000

train_dataset, validation_dataset, _ = torch.utils.data.random_split(
    cifar10_train_data, [train_size, test_size, ignored_train_data_size]
)

batch_size = config.batch_size

train_dataloader = DataLoader(train_dataset, batch_size)
validation_dataloader = DataLoader(validation_dataset, batch_size)
test_dataloader = DataLoader(cifar10_test_data, batch_size)

network = network.to("cuda")

# optimiser + loss function
sgd = optim.SGD(network.parameters(), lr=config.lr, momentum=config.momentum)
criterium = nn.CrossEntropyLoss()

# train for 20 epochs
train_errs, valid_errs = [], []
for epoch_idx in trange(20):
    local_errs = update(network, train_dataloader, criterium, sgd)
    train_errs.append(sum(local_errs) / len(local_errs))
    validation_error_epoch = evaluate(network, validation_dataloader, criterium)
    valid_errs.append(sum(validation_error_epoch) / len(validation_error_epoch))
    wandb.log({"validation_loss":sum(validation_error_epoch) / len(validation_error_epoch)}, step=epoch_idx)

# plot learning curves
# from matplotlib import pyplot as plt
#
# plt.plot(train_errs, label="train")
# plt.plot(valid_errs, label="valid")
# plt.legend()
# plt.show()


# print(f"ran on {next(network.parameters().device)}")

@torch.no_grad()
def accuracy(logits, targets):
    """
    Compute the accuracy for given logits and targets.

    Parameters
    ----------
    logits : (N, K) torch.Tensor
        A mini-batch of logit vectors from the network.
    targets : (N, ) torch.Tensor
        A mini_batch of target scalars representing the labels.

    Returns
    -------
    acc : () torch.Tensor
        The accuracy over the mini-batch of samples.
    """

    tot_correct = tot_samples = 0

    _, predictions = logits.max(1)
    tot_correct += (predictions == targets).sum()
    tot_samples += predictions.size(0)

    accuracy_samples = (tot_correct.item() / tot_samples) * 100

    return accuracy_samples


network.eval()
test_accuracy = []
for batch_idx, (inputs, targets) in enumerate(test_dataloader):
    inputs = inputs.to("cuda")
    targets = targets.to("cuda")

    logits = network(inputs)
    test_accuracy_mini_batch = accuracy(logits, targets)
    test_accuracy.append(test_accuracy_mini_batch)

print(sum(test_accuracy) / len(test_accuracy))
