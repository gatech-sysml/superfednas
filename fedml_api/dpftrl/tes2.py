#define the network in pytorch (with Z)
import torch.nn as nn
import torch
from torchvision import datasets, transforms

from dp_ftrl import FTRLM

class NeuralNetworkZ(nn.Module):
    def __init__(self, input_dim, hidden_dim, lr_w):
        super(NeuralNetworkZ, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim[0])
        self.lin2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.lin3 = nn.Linear(hidden_dim[1], hidden_dim[2])

        self.apply(self.init_weights_biases)

        self.optimizer_w = torch.optim.Adam(self.get_w(), lr=lr_w)

        self.criterion_z = nn.MSELoss()

    def init_weights_biases(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.0) #modify with custom function later
            nn.init.normal_(m.bias, mean=0.0, std=1.0)

    def get_w(self):
        return [self.input_layer.weight, self.lin2.weight, self.lin3.weight]

    def forward(self, x, training=True):
        x = self.input_layer(x)
        x = nn.Tanh()(x)
        x = self.lin2(x)
        x = nn.Tanh()(x)
        x = self.lin3(x)
        return x

    def compute_ce_loss(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss()
        final_loss = ce_loss(logits, targets)
        return final_loss


    def loss(self, x, targets):
        logits = self.forward(x)
        final_loss = self.compute_ce_loss(logits, targets)
        return final_loss
        

if __name__ == '__main__':
    input_dim = 784

    #Parameters setup
    classes = 10
    lr_w = 0.001
    batch_size = 100
    hidden_dim = [100, 100, 10]
    epochs = 3

    #Get MNIST data and define transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
    trainset = datasets.MNIST('mnist_train', download=True, train=True, transform=transform)
    testset = datasets.MNIST('mnist_test', download=True, train=False, transform=transform)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    model = NeuralNetworkZ(input_dim, hidden_dim, lr_w)

    bs = 1000
    noise_multiplier = 7.0
    print("#############params###############")
    param_list = []
    param_shapes = [p.shape for p in model.parameters()]
    for p in model.parameters():
        param_list.append(p)
    print("#############params###############")


    optimizer_FTRLM = FTRLM( # noqa
            device="cpu",
            params=param_list,
            model_param_sizes=param_shapes,
            lr=0.01,
            momentum=0.9,
            nesterov=True,
            noise_std=(noise_multiplier / bs),
            max_grad_norm=1.0,
            seed=0,
            efficient=True,
            )
    
    model.optimizer_w = optimizer_FTRLM

    criterion = nn.CrossEntropyLoss()

    device = 'cpu'
    n_total_steps = len(trainloader)

    beta = 10
    z_iter = 100

    for epoch in range(epochs):
        #reinit Z here every epoch

        for i, (images, labels) in enumerate(trainloader):
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            # Forward pass
            #outputs = model(images)
            total_loss = model.loss(images, labels)
            # Backward and optimize
            model.optimizer_w.zero_grad()
            #loss.backward()
            total_loss.backward()
            model.optimizer_w.step()

            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{epochs}], Step[{i+1}/{n_total_steps}], Loss: {total_loss.item():.8f}, CE Loss: {total_loss.item():.10f}')

                with torch.no_grad():
                    n_correct = 0
                    n_samples = 0
                    for images, labels in testloader:
                        images = images.reshape(-1, 28*28).to(device)
                        labels = labels.to(device)
                        outputs = model(images, training=False)
                        # max returns (value ,index)
                        _, predicted = torch.max(outputs.data, 1)
                        n_samples += labels.size(0)
                        n_correct += (predicted == labels).sum().item()

                acc = 100.0 * n_correct / n_samples
                print(f'Accuracy of the network on the 10000 test images: {acc} %')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in testloader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images, training=False)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')