import numpy as np
import os
import torch
from time import time, gmtime, strftime
import torch.nn as nn
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import RandomCrop, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt
#from tensorboardX import SummaryWriter


def ourTrain(model, train_loader,val_loader, optimizer, loss_fn, scheduler, saveWeights=False, saving_path="", loadWeights=False, loading_path="", device=torch.device('cpu'), print_every=100, self_train=False, n_epochs=20, logfiles="", folder_name=''):
    loading_path = folder_name+loading_path
    saving_path = folder_name+saving_path
    logfiles = folder_name+logfiles
    # create summary writer for tensorboard
    #writer = SummaryWriter(logfiles)
    if loadWeights:
        model.load_state_dict(torch.load(loading_path+'.pth'), strict=False)
        model.to(device)
        print("model weights are loaded from ", device)
    best_loss = 999999999999999999999
    #model.train()
    train_losses = []
    val_losses = []
    train_accuracies_1, train_accuracies_3, train_accuracies_5 = [], [], []
    val_accuracies_1, val_accuracies_3, val_accuracies_5 = [], [] , []
    t1=time()
    n_it = int(len(train_loader.dataset)/train_loader.batch_size)
    n_it_val = int(len(val_loader.dataset)/val_loader.batch_size)

    final_prediction_train = np.zeros((len(train_loader.dataset),3))
    final_prediction_val = np.zeros((len(val_loader.dataset), 3))

    for epoch in range(n_epochs):
        t5 = time()
        losses = []
        n_correct_1, n_correct_3, n_correct_5 = 0, 0, 0
        # Training
        model.train()

        for iteration, (images, labels, idx) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)
            idx = idx.to(device)
            output = model(images)
            optimizer.zero_grad()
            if self_train:
                loss = loss_fn(output, labels.float())
            else:
                loss = loss_fn(output, labels.long())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if iteration % print_every == 0:

                t2=time()
                est_ep = (t2-t5)*(n_it-iteration)/(iteration+1)
                est_end = (t2-t1)*(n_it*n_epochs-iteration-epoch*n_it+(n_epochs-epoch)*n_it_val)/(iteration+1+epoch*n_it+epoch*n_it_val)
                print("Iteration: " + str(iteration) + " of " + str(n_it) + "   time until end of epoch: " + strftime("%H:%M:%S", gmtime(est_ep)) + '   end of traning in ' + strftime("%H:%M:%S", gmtime(est_end)))

            if self_train==False:
                n_correct_1 += torch.sum(output.argmax(1) == labels).item()
                top3_probs, top3_labs = torch.topk(output, 3)
                top5_probs, top5_labs = torch.topk(output, 5)

                for i in range(len(labels)):
                    l = labels[i].item()
                    # print("l: ", l)
                    if l in top3_labs[i]:
                        n_correct_3 += 1
                    if l in top5_labs[i]:
                        n_correct_5 += 1

                # Save the predictions of last epoch
                if epoch == n_epochs - 1:
                    first_ind = iteration*train_loader.batch_size
                    if len(train_loader.dataset)-first_ind < train_loader.batch_size:
                        last_ind = len(train_loader.dataset)
                    else:
                        last_ind = iteration*train_loader.batch_size + train_loader.batch_size

                    final_prediction_train[first_ind:last_ind, 0] = idx.tolist()
                    final_prediction_train[first_ind:last_ind, 1] = labels.tolist()
                    final_prediction_train[first_ind:last_ind, 2] = output.argmax(1).tolist()
        if self_train == False:
            np.savetxt(saving_path + '_final_train_res.txt', final_prediction_train, fmt='%s')
        curr_loss = np.mean(np.array(losses))
        #writer.add_scalar('Train/Loss', curr_loss, epoch)

        print('Loss after epoch '+str(epoch+1)+'/'+str(n_epochs)+' is:  '+str(curr_loss))
        if self_train==False:
            accuracy_1 = 100.0 * n_correct_1 / len(train_loader.dataset)
            train_accuracies_1.append(accuracy_1)

            accuracy_3 = 100.0 * n_correct_3 / len(train_loader.dataset)
            train_accuracies_3.append(accuracy_3)

            accuracy_5 = 100.0 * n_correct_5 / len(train_loader.dataset)
            train_accuracies_5.append(accuracy_5)

            print('Top-1 Accuracy after epoch '+str(epoch+1)+'/'+str(n_epochs)+' is:  '+str(accuracy_1))
            print('Top-3 Accuracy after epoch ' + str(epoch + 1) + '/' + str(n_epochs) + ' is:  ' + str(accuracy_3))
            print('Top-5 Accuracy after epoch ' + str(epoch + 1) + '/' + str(n_epochs) + ' is:  ' + str(accuracy_5))
            #writer.add_scalar('Train/Acc', accuracy, epoch)
        train_losses.append(curr_loss)
        losses=[]
        n_correct_1, n_correct_3, n_correct_5 = 0, 0, 0
        t3 = time()
        print('Start validation')
        model.eval()
        #Validation
        with torch.no_grad():
            for iteration, (images, labels, idx) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                idx = idx.to(device)
                output = model(images)
                optimizer.zero_grad()
                if self_train:
                    loss = loss_fn(output, labels.float())
                else:
                    loss = loss_fn(output, labels.long())
                losses.append(loss.item())
                if iteration % print_every == 0:
                    t4 = time()
                    est_ep = (t4 - t3) * (n_it_val - iteration) / (iteration + 1)
                    print("Iteration: " + str(iteration) + " of " + str(n_it_val) + "    time until end of validation: " + strftime(
                        "%H:%M:%S", gmtime(est_ep)))
                if self_train == False:
                    n_correct_1 += torch.sum(output.argmax(1) == labels).item()
                    top3_probs, top3_labs = torch.topk(output, 3)
                    top5_probs, top5_labs = torch.topk(output, 5)

                    for i in range(len(labels)):
                        l = labels[i].item()
                        # print("l: ", l)
                        if l in top3_labs[i]:
                            n_correct_3 += 1
                        if l in top5_labs[i]:
                            n_correct_5 += 1

                    # Save the predictions of last epoch
                    if epoch == n_epochs - 1:

                        first_ind = iteration * val_loader.batch_size
                        if len(val_loader.dataset) - first_ind < val_loader.batch_size:
                            last_ind = len(val_loader.dataset)
                        else:
                            last_ind = iteration * val_loader.batch_size + val_loader.batch_size

                        final_prediction_val[first_ind:last_ind, 0] = idx.tolist()
                        final_prediction_val[first_ind:last_ind, 1] = labels.tolist()
                        final_prediction_val[first_ind:last_ind, 2] = output.argmax(1).tolist()

            if self_train == False:
                np.savetxt(saving_path+'_final_val_res.txt', final_prediction_val, fmt='%s')

        curr_val_loss = np.mean(np.array(losses))
        val_losses.append(curr_val_loss)
        #writer.add_scalar('Val/Loss', curr_val_loss, epoch)

        if val_losses[-1]<best_loss and saveWeights:
            best_loss=val_losses[-1]
            print("Model weights are saved on (autosave)", device)
            torch.save(model.state_dict(), saving_path+'_autosave_'+'.pth')

        print('Loss at validation ' + str(epoch + 1) + ' is:  ' + str(curr_loss))
        if self_train==False:
            accuracy_1 = 100.0 * n_correct_1 / len(val_loader.dataset)
            val_accuracies_1.append(accuracy_1)

            accuracy_3 = 100.0 * n_correct_3 / len(val_loader.dataset)
            val_accuracies_3.append(accuracy_3)

            accuracy_5 = 100.0 * n_correct_5 / len(val_loader.dataset)
            val_accuracies_5.append(accuracy_5)
            print('Top-1 Accuracy after validation:  ' + str(accuracy_1))
            print('Top-3 Accuracy after validation:  ' + str(accuracy_3))
            print('Top-5  Accuracy after validation:  ' + str(accuracy_5))
            #writer.add_scalar('Val/Acc', accuracy, epoch)
        scheduler.step()

    if saveWeights:
        print("Model weights are saved on ", device)
        torch.save(model.state_dict(), saving_path+'.pth')

    #writer.close()
    return train_losses, val_losses, train_accuracies_1, train_accuracies_3, train_accuracies_5, val_accuracies_1, val_accuracies_3, val_accuracies_5


def classificationTest(model, test_loader, device=torch.device('cpu'), print_every=100, loading_path='', folder_name=''):
    loading_path = folder_name+loading_path
    model.load_state_dict(torch.load(loading_path+'.pth'), strict=False)
    model.to(device)
    
    print("model weights are loaded from ", device)
    test_accuracies_1, test_accuracies_3, test_accuracies_5 = [], [], []
    n_correct_1, n_correct_3, n_correct_5 = 0, 0, 0
    final_prediction_test = np.zeros((len(test_loader.dataset),3))
    t1=time()
    n_it = int(len(test_loader.dataset)/test_loader.batch_size)
    model.eval()
    with torch.no_grad():
        for iteration, (images, labels, idx) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            idx = idx.to(device)
            output = model(images)
            if iteration % print_every == 0:
                t2 = time()
                est_ep = (t2 - t1) * (n_it - iteration) / (iteration + 1)
                print("Iteration: " + str(iteration) + " of " + str(
                    n_it) + "    time until end of test: " + strftime(
                    "%H:%M:%S", gmtime(est_ep)))

            n_correct_1 += torch.sum(output.argmax(1) == labels).item()
            top3_probs, top3_labs = torch.topk(output, 3)
            top5_probs, top5_labs = torch.topk(output, 5)

            for i in range(len(labels)):
                l = labels[i].item()
                # print("l: ", l)
                if l in top3_labs[i]:
                    n_correct_3 += 1
                if l in top5_labs[i]:
                    n_correct_5 += 1

            first_ind = iteration * test_loader.batch_size
            if len(test_loader.dataset) - first_ind < test_loader.batch_size:
                last_ind = len(test_loader.dataset)
            else:
                last_ind = iteration * test_loader.batch_size + test_loader.batch_size

            final_prediction_test[first_ind:last_ind, 0] = idx.tolist()
            final_prediction_test[first_ind:last_ind, 1] = labels.tolist()
            final_prediction_test[first_ind:last_ind, 2] = output.argmax(1).tolist()

        np.savetxt(loading_path + '_final_test_res.txt', final_prediction_test, fmt='%s')

        accuracy_1 = 100.0 * n_correct_1 / len(test_loader.dataset)
        test_accuracies_1.append(accuracy_1)
        accuracy_3 = 100.0 * n_correct_3 / len(test_loader.dataset)
        test_accuracies_3.append(accuracy_3)
        accuracy_5 = 100.0 * n_correct_5 / len(test_loader.dataset)
        test_accuracies_5.append(accuracy_5)

        print('Top-1 Accuracy:  ' + str(accuracy_1))
        print('Top-3 Accuracy:  ' + str(accuracy_3))
        print('Top-5 Accuracy:  ' + str(accuracy_5))

    return test_accuracies_1, test_accuracies_3, test_accuracies_5


def plotCompare(image_idx, dataset):
    images=[]
    dir = dataset.root_dir
    for r, d, f in os.walk(dir):
        for file in f:
            images.append(os.path.join(dir, file))
    image = Image.open(images[image_idx])
    image.show()
    per_im, lab = dataset[image_idx]
    transforms.ToPILImage()(per_im).show()
    tile_size = int(np.sqrt(lab.shape[0]))
    lab_a = np.zeros((tile_size,tile_size))
    for i in range(tile_size):
        #k, l = divmod(i, tile_size)
        lab_a[i] = lab[i*tile_size:i*tile_size+tile_size]
    print(lab_a)

def plot_loss(n_epochs, train_losses, val_losses):
    plt.figure()
    plt.plot(np.arange(n_epochs), train_losses)
    plt.plot(np.arange(n_epochs), val_losses)
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('Train/val loss');

def plot_acc(n_epochs, train_acc, val_acc):
    plt.figure()
    plt.plot(np.arange(n_epochs), train_acc)
    plt.plot(np.arange(n_epochs), val_acc)
    plt.legend(['train_accuracies', 'val_accuracies'])
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('Train/val accuracies');
