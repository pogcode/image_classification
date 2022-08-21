"""
    描述：LeNet的train脚本
    时间：2022/08/21
    作者：pogcode
    参考：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
"""
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from models.LeNet.LeNet import LeNet


def main():

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR10共50000张训练图片
    train_dataset = torchvision.datasets.CIFAR10(root='../../datasets', train=True,
                                             download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    # CIFAR10共10000张验证图片
    val_dataset = torchvision.datasets.CIFAR10(root='../../datasets', train=False,
                                               download=True, transform=transform)

    val_dataloader = DataLoader(val_dataset, batch_size=5000, shuffle=False, num_workers=0)

    val_data_iter = iter(val_dataloader)
    val_image, val_label = val_data_iter.next()

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for step, data in enumerate(train_dataloader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:  # print every 500 mini-batches
                with torch.no_grad():
                    outputs = net(val_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()