import numpy as np
import torch



def train(model, train_loader, optimizer, loss_fn, saveWeights, saving_path, loadWeights, loading_path, device, print_every=100):

    if loadWeights:
        model.load_state_dict(torch.load(loading_path), strict=False)
        model.to(device)
        print("model weights are loaded from ", device)

    model.train()

    losses = []
    n_correct = 0
    for iteration, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        optimizer.zero_grad()
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        #         if iteration % print_every == 0:
        #             print('Training iteration {}: loss {:.4f}'.format(iteration, loss.item()))
        losses.append(loss.item())
        n_correct += torch.sum(output.argmax(1) == labels).item()
    accuracy = 100.0 * n_correct / len(train_loader.dataset)

    if saveWeights:
        print("Model weights are saved on ", device)
        torch.save(model.state_dict(), saving_path)

    return np.mean(np.array(losses)), accuracy


def test(model, test_loader, loss_fn, device):

    model.eval()
    test_loss = 0
    n_correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_fn(output, labels)
            test_loss += loss.item()
            n_correct += torch.sum(output.argmax(1) == labels).item()

    average_loss = test_loss / len(test_loader)
    accuracy = 100.0 * n_correct / len(test_loader.dataset)
    return average_loss, accuracy


def fit(train_dataloader, val_dataloader, model, optimizer, loss_fn, n_epochs, device, scheduler=None, saveWeights=False, saving_path=None, loadWeights=False, loading_path=None):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(n_epochs):
        print("epoch: ", epoch)
        train_loss, train_accuracy = train(model, train_dataloader, optimizer, loss_fn, saveWeights, saving_path, loadWeights, loading_path, device)
        val_loss, val_accuracy = test(model, val_dataloader, loss_fn, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        if scheduler:
            scheduler.step()  # argument only needed for ReduceLROnPlateau
        print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(
            epoch + 1, n_epochs,
            train_losses[-1],
            train_accuracies[-1],
            val_losses[-1],
            val_accuracies[-1]))

    return train_losses, train_accuracies, val_losses, val_accuracies


