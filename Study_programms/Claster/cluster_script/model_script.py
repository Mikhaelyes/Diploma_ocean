import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
from matplotlib import pyplot as plt


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_mnist_data = MNIST(
    ".", train=True, transform=torchvision.transforms.ToTensor(), download=True
)


test_mnist_data = MNIST(
    ".", train=False, transform=torchvision.transforms.ToTensor(), download=True
)

train_data_loader = torch.utils.data.DataLoader(
    train_mnist_data, batch_size=32, shuffle=True, num_workers=2
)

test_data_loader = torch.utils.data.DataLoader(
    test_mnist_data, batch_size=32, shuffle=False, num_workers=2
)


activation = nn.Mish


#Encoder
model = nn.Sequential()
model.add_module('l1_e', nn.Conv2d(1, 4, kernel_size=3, padding=(1, 1), stride=(2, 2)))
model.add_module('activation1', activation())
model.add_module('l2_e', nn.Conv2d(4, 16, kernel_size=3, padding=(1, 1), stride=(2, 2)))
model.add_module('activation2', activation())
model.add_module('l3_e', nn.Conv2d(16, 64, kernel_size=3, padding=(0, 0), stride=(1, 1)))
model.add_module('activation3', activation())
model.add_module('l4_e', nn.Conv2d(64, 256, kernel_size=3, padding=(0, 0), stride=(1, 1)))
model.add_module('activation4', activation())
model.add_module('l5_e', nn.Conv2d(256, 1024, kernel_size=3, padding=(0, 0), stride=(1, 1)))
model.add_module('activation5', activation())


model.add_module('l6_d', nn.ConvTranspose2d(1024, 256, kernel_size=3))
model.add_module('activation6', activation())
model.add_module('l7_d', nn.ConvTranspose2d(256, 64, kernel_size=3))
model.add_module('activation7', activation())
model.add_module('l8_d', nn.ConvTranspose2d(64, 16, kernel_size=3))
model.add_module('activation8', activation())
model.add_module('l9_e', nn.ConvTranspose2d(16, 4, kernel_size=4, padding=(1, 1), stride=(2, 2), dilation=(1, 1)))
model.add_module('activation9', activation())
model.add_module('l10_e', nn.ConvTranspose2d(4, 1, kernel_size=4, padding=(1, 1), stride=(2, 2), dilation=(1, 1)))


# model.add_module('l3', nn.Conv2d(15, 50, kernel_size=3, padding=(1, 1)))


opt = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 1
loss_func = nn.MSELoss(reduction='mean')


def train_model(model, train_loader, loss_fn, opt, n_epochs: int):
    train_loss = []
    val_loss = []
    val_accuracy = []
    input_im = []
    output_im = []
    
    for epoch in tqdm(range(n_epochs)):
        ep_train_loss = []
        ep_val_loss = []
        ep_val_accuracy = []

        model.to(device)
        model.train(True) # enable dropout / batch_norm training behavior
        i = 0
        for X_batch, y_batch in tqdm(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # move data to target device
            ### YOUR CODE HERE

            # train on batch: compute loss, calc grads, perform optimizer step and zero the grads
            ### YOUR CODE HERE
            # print(X_batch.reshape(-1, 1, 28, 28).shape)
            y_pred = model(X_batch.reshape(-1, 1, 28, 28))
            # print(y_pred.shape)
            # print(y_pred.shape)
            loss = loss_func(y_pred, X_batch.reshape(-1, 1, 28, 28))
            loss.backward()
            opt.step()
            opt.zero_grad()
            ep_train_loss.append(loss.item())

            if ((epoch == 0) and (i == 0)):
                input_im.append(X_batch.reshape(-1, 1, 28, 28))
                output_im.append(y_pred)

            i += 1
                
        model.train(False) # disable dropout / use averages for batch_norm
            


        train_loss.append(np.mean(ep_train_loss))
        # print(train_loss)
        # val_loss.append(np.mean(ep_val_loss))
        # val_accuracy.append(np.mean(ep_val_accuracy))

    return model, train_loss, input_im, output_im


if True:
    model, train_loss, input_im, output_im = train_model(model, train_data_loader, loss_func, opt, n_epochs)
    # torch.save(model.state_dict(), 'model_weights_cnn_1.pth')

if False:
    model.load_state_dict(torch.load('model_weights_cnn_1.pth'))


if True:
    test_im = []
    for X_batch, y_batch in tqdm(train_data_loader):
        test_im.append(X_batch.reshape(-1, 1, 28, 28))
     
    for i in range(5):
        y_pred = model(test_im[0][i])

        im_x = test_im[0][i].detach().numpy()
        im_y = y_pred.detach().numpy()        
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(im_x.reshape(28, 28))       
        ax[1].imshow(im_y.reshape(28, 28))

        fig.savefig('test'+ str(i) +'.png')
