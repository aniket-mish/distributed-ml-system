from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def get_dataset():
    """
    Get the data
    """
    training_data = datasets.CIFAR10(
        root="data", train=True, download=True, transform=ToTensor()
    )

    test_data = datasets.CIFAR10(
        root="data", train=False, download=True, transform=ToTensor()
    )

    print(f"We have {len(training_data)} examples in the train set")
    print(f"We have {len(test_data)} examples in the test set")

    class_names = training_data.classes

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    return train_dataloader, test_dataloader, class_names
