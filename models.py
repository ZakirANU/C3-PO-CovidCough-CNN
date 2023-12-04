import os
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from utils.log import Log
from datetime import datetime
from sklearn.model_selection import train_test_split



class CovidDataset(Dataset):
    """
    A custom pytorch dataset used to load Covid data.

    :param X: Feature map (n_samples, n_features)
    :param y: labels (n_samples,)
    """
    
    def __init__(self, X, y, n_class, transform = None):
        """
        @type X: numpy.ndarray
        @type y: numpy.ndarray
        """
        
        self.data = X
        self.label = y
        self.n_class = n_class
        self.transform = transform

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx]
        label = self.label[idx]
        
        if self.transform:
            data = self.transform(data)

        return data, label


class CovidCNNModel(Module):
    """
    A custom Pytorch module, used to diagnose COVID-19
    from cough signals. 
    Adaptive to different number of input features.
    """

    def __init__(self, dataset):
        """
        @type dataset: CovidDataset
        """

        super().__init__()

        sample = next(iter(dataset))[0]
        if not isinstance(sample, torch.Tensor):
            sample = torch.Tensor(sample)

        self.backbone = nn.Sequential(
            nn.Conv2d(sample.shape[0], 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten()
        )

        sample = self.backbone(sample.unsqueeze(0))

        self.linear = nn.Sequential(
            nn.Linear(sample.shape[1], 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, dataset.n_class),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.linear(x)
        return x

class CovidCNNModel2(Module):
    """
    Another custom Pytorch module, used to diagnose COVID-19
    from cough signals. 
    Adaptive to different number of input features.
    """

    def __init__(self, dataset):
        """
        @type dataset: CovidDataset
        """

        super().__init__()

        sample = next(iter(dataset))[0]
        if not isinstance(sample, torch.Tensor):
            sample = torch.Tensor(sample)

        self.backbone = nn.Sequential(
            nn.Conv2d(sample.shape[0], 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        sample = self.backbone(sample.unsqueeze(0))

        self.linear = nn.Sequential(
            nn.Linear(sample.shape[1], 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, dataset.n_class),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.linear(x)
        return x


def numberOfParam(model: Module):
    """
    Calculate the number of parameters in a model
    """
    param_num = 0
    for param in model.parameters():
        param_num += param.numel()

    print(param_num)


def testModel(model: Module, testloader, 
        criterion = nn.CrossEntropyLoss(), analyze: bool = False):
    """
    Test a model using the given testlaoder.

    :param model: Test model
    :param testloader: Test dataset loader
    :param criterion: Loss function
    :param analyze: Whether need to output 
                    predicted y and ground truth

    :return: Out put accuracy and loss.
             Also output predicted y and ground truth for
             analysing if "analyze" is true.
    """

    criterion.reduction = "sum"
    
    # Get device
    device = next(model.parameters()).device

    # Store the previous model mode (training or evaluation)
    was_training = model.training
    # Set the mode to evaluation mode
    model.train(False)

    # Number of samples predicted correctly
    correct_sum = 0
    # Total number of samples
    total = 0
    # Sum of losses
    loss_sum = 0

    y_out = None
    y_true = None

    softmax = nn.Softmax(dim = 1)

    with torch.no_grad():
        for data in testloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device).type(torch.long)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()

            # validate
            preds = torch.argmax(outputs, dim = 1)
            correct = (preds == labels).sum().item()
            correct_sum += correct
            total += labels.size(0)

            # Record all outputs and the corresponding labels
            if analyze:

                if y_out is None:
                    y_out = np.empty([0, outputs.size()[1]])
                if y_true is None:
                    y_true = np.empty([0])

                y_out = np.append(y_out, softmax(outputs).cpu().numpy(), axis = 0)
                y_true = np.append(y_true, labels.cpu().numpy())

    accuracy = correct_sum / total
    loss_avg = loss_sum / total

    # Restore model mode
    model.train(was_training)

    if analyze:
        return accuracy, loss_avg, y_out, y_true 
    else:
        return accuracy, loss_avg


def train(model: Module, *, trainloader: DataLoader, valloader: DataLoader, epoch_num: int,
    record_freq: int, criterion = nn.CrossEntropyLoss(), optimizer: Optimizer = None, 
    out_dir: str = "temp"):

    """
    Train a given model with trainloader and valloader. Will reocrd
    the accuracy and loss during training, and save the file in out_dir.
    
    :param model: All parameters in model must be in a same device.
    """
    criterion.reduction = "sum"

    # Create an output directory with a timestamp
    out_dir = os.path.join(out_dir, f"{type(model).__name__}_{timestamp()}")
    while os.path.exists(out_dir):
        out_dir = out_dir + "_new"
    os.mkdir(out_dir)

    # Create log
    report_file_path = os.path.join(out_dir, "report.out")
    log = Log(report_file_path)

    # Store the previous model mode (training or evaluation)
    was_training = model.training
    # Set the mode to training mode
    model.train(True)

    # Specify a default optimizer
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, betas = (0.9, 0.999))

    # Get device
    device = next(model.parameters()).device

    # Record the accuracy and loss of training and validation every {record_freq} batches
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    # The best model
    best_model = None

    # Timekeeper
    starttime = time()
    timer = starttime

    # loop over the dataset multiple times
    for i_epoch in range(epoch_num):

        # Number of images predicted correctly
        correct_sum = 0
        # Total number of images
        total = 0
        # Sum of loss
        loss_sum = 0
        
        for i_batch, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device).type(torch.long)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            # Validate
            preds = torch.argmax(outputs, dim = 1)

            # Stats
            correct = (preds == labels).sum().item()
            correct_sum += correct
            total += labels.size(0)
            loss_sum += loss.item()

            # Show result
            if (i_batch + 1) % record_freq == 0:
                
                # Calculate the accuracy and loss of training and validation
                train_accuracy = correct_sum / total
                train_loss = loss_sum / total
                val_accuracy, val_loss = testModel(model, valloader, criterion)

                # Save model
                if len(val_accuracies) == 0 or (len(val_accuracies) > 0 and max(val_accuracies) < val_accuracy):
                    best_model = f"[epoch:{i_epoch + 1}, batch:{i_batch + 1:3d}]"
                    torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
                
                # Record
                train_accuracies.append(train_accuracy)
                train_losses.append(train_loss)
                val_accuracies.append(val_accuracy)
                val_losses.append(val_loss)

                # Print
                report = f'[epoch:{i_epoch + 1}, batch:{i_batch + 1:3d}] training loss: {train_loss:.4f}, '\
                    f'training accuracy: {train_accuracy:.3f}, validation loss: {val_loss:.4f}, '\
                    f'validation accuracy: {val_accuracy:.3f}, time used: {time() - timer: 3.1f} s, '\
                    f'total time used: {time() - starttime: 4.1f} s.'
                log.log(report)

                
                # Number of images predicted correctly
                correct_sum = 0
                # Total number of images
                total = 0
                # Sum of loss
                loss_sum = 0
                

    # Save accuracies and losses
    np.save(os.path.join(out_dir, "train_accuracies.npy"), np.asarray(train_accuracies))
    np.save(os.path.join(out_dir, "train_losses.npy"), np.asarray(train_losses))
    np.save(os.path.join(out_dir, "val_accuracies.npy"), np.asarray(val_accuracies))
    np.save(os.path.join(out_dir, "val_losses.npy"), np.asarray(val_losses))

    report = f'Training Finished. Total running time is {(time() - starttime) / 60: 3.2f} min\n' + \
        f'The best model: {best_model}, the best validation accuracy: {max(val_accuracies)}, the best validation loss: {min(val_losses)}.'
    log.log(report)

    # Restore model mode
    model.train(was_training)

    return train_accuracies, train_losses, val_accuracies, val_losses, out_dir


def plotTrainingProcess(record_freq, data_loader, train_accuracies, train_losses, val_accuracies, val_losses, out_dir = None):
    """
    Plot training process using data recorded by `train` function.
    """


    n_record = len(train_accuracies)
    plot_x = np.arange(n_record) / (len(data_loader) // record_freq)

    plt.figure(figsize=(15,4), dpi = 150)
    
    plt.subplot(121)
    plt.plot(plot_x, train_accuracies)
    plt.plot(plot_x, val_accuracies)

    plt.legend(["training accuracy", "validation accuracy"])
    plt.xlabel("Epoches")
    plt.ylabel("Accuracy")

    plt.subplot(122)
    plt.plot(plot_x, train_losses)
    plt.plot(plot_x, val_losses)

    plt.legend(["training loss", "validation loss"])
    plt.xlabel("Epoches")
    plt.ylabel("Loss")

    if out_dir:
        plt.savefig(os.path.join(out_dir, "training_process.jpg"), bbox_inches = 'tight')


def testPerformance(model, testloader, class_names = None, out_dir = None):
    """
    Test and calculate the performance of the model, can save ROC curve
    and Output histogram in out_dir.
    """
    
    report_path = os.path.join(out_dir, "report.out") if out_dir is not None else None
    log = Log(report_path)

    accuracy, loss, y_out, y_true = testModel(model, testloader, analyze = True)

    report = f"[test performance] Test accuracy: {accuracy:.3f}, Test loss: {loss:.4f}"
    log.log(report)

    n_class = y_out.shape[1]

    aucs = np.zeros(n_class)

    for class_i in range(n_class):

        y_out_i = y_out[:, class_i]
        y_true_i = (y_true == class_i).astype(np.int8)
        if class_names:
            name = class_names[class_i]
        else:
            name = str(class_i)

        fpr, tpr, thresholds = roc_curve(y_true_i, y_out_i)
        auc = roc_auc_score(y_true_i, y_out_i)
        aucs[class_i] = auc

        report = f"[class: {name}] AUC: {auc}"
        log.log(report)

        plt.figure(figsize = (15, 4), dpi = 150)
        # plt.suptitle(f"class: {name}")
        plt.subplot(121)
        plt.plot(fpr, tpr, c = "C0", lw = 2)
        plt.plot([0, 1], [0, 1], ls = "--", c = "C2", lw = 2)
        plt.title("ROC curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")

        plt.subplot(122)
        _ = plt.hist(y_out_i, bins = 100)
        plt.xlabel("output")
        plt.ylabel("count")
        plt.title("Outputs histogram")
            
        if out_dir:
            plt.savefig(os.path.join(out_dir, f"performance_{name}.jpg"), bbox_inches = 'tight')

    return accuracy, loss, aucs


def cross_validation(module: type, X: np.ndarray, y: np.ndarray, batch_size: int, epoch_num: int, out_dir: str = "temp"):
    """
    Regular cross validation
    """

    # Need to be updated, refer to cross_validation_test

    # Create an output directory with a timestamp
    out_dir = os.path.join(out_dir, f"{module.__name__}_CV_{timestamp()}")
    while os.path.exists(out_dir):
        out_dir = out_dir + "_new"
    os.mkdir(out_dir)

    # Use GPU to train and infer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    # Reshape to (N, C, H, W), where C = 1
    X = X[:, None, :, :]

    # K-fold
    n_splits = 5
    k_fold = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 7)
    train_val_indexs = k_fold.split(X, y)
    kf_accuracies = []
    kf_losses = []
    kf_aucs = []

    for train_index, val_index in train_val_indexs:

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        n_class = np.unique(y).shape[0]
        record_freq = 1

        trainset = CovidDataset(X_train, y_train, n_class)
        trainloader = DataLoader(trainset, batch_size, shuffle = True, num_workers = 0)

        valset = CovidDataset(X_val, y_val, n_class)
        valloader = DataLoader(valset, batch_size, shuffle = True, num_workers = 0)

        model = module(trainset)
        model.to(device)

        train_accuracies, train_losses, val_accuracies, val_losses, stamped_out_dir = train(
            model = model, 
            trainloader = trainloader, 
            valloader = valloader, 
            epoch_num = epoch_num,
            record_freq = record_freq, 
            out_dir = out_dir)

        plotTrainingProcess(
            record_freq, 
            trainloader, 
            train_accuracies, 
            train_losses, 
            val_accuracies, 
            val_losses,
            stamped_out_dir
        )

        model.load_state_dict(torch.load(os.path.join(stamped_out_dir, "best_model.pth")))

        accuracy, loss, aucs = testPerformance(model, valloader, out_dir = stamped_out_dir)
        auc = np.average(aucs)

        kf_accuracies.append(accuracy)
        kf_losses.append(loss)
        kf_aucs.append(auc)

    # Close opened matplotlib figures
    plt.close("all")

    kf_accuracy = sum(kf_accuracies) / len(kf_accuracies)
    kf_loss = sum(kf_losses) / len(kf_losses)
    kf_auc = sum(kf_aucs) / len(kf_aucs)

    return kf_accuracy, kf_loss, kf_auc

def cross_validation_test(module: type, X: np.ndarray, y: np.ndarray, batch_size: int, epoch_num: int, out_dir: str = "temp"):
    """
    Nested Cross validation with the number of epoch to early stop 
    as hyperparamer.
    """


    # Create an output directory with a timestamp
    out_dir = os.path.join(out_dir, f"{module.__name__}_CVT_{timestamp()}")
    while os.path.exists(out_dir):
        out_dir = out_dir + "_new"
    os.mkdir(out_dir)

    # Create log
    report_file_path = os.path.join(out_dir, "report.out")
    log = Log(report_file_path)

    # Use GPU to train and infer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    # Reshape to (N, C, H, W), where C = 1
    X = X[:, None, :, :]

    # K-fold
    n_splits = 5
    k_fold = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 7)
    train_valtest_indexs = k_fold.split(X, y)
    kf_accuracies = []
    kf_losses = []
    kf_aucs = []

    for train_index, valtest_index in train_valtest_indexs:

        X_train, X_valtest = X[train_index], X[valtest_index]
        y_train, y_valtest = y[train_index], y[valtest_index]
        X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, stratify=y_valtest, random_state=7)

        n_class = np.unique(y).shape[0]
        record_freq = 1

        trainset = CovidDataset(X_train, y_train, n_class)
        trainloader = DataLoader(trainset, batch_size, shuffle = True, num_workers = 0)

        valset = CovidDataset(X_val, y_val, n_class)
        valloader = DataLoader(valset, batch_size, shuffle = True, num_workers = 0)

        testset = CovidDataset(X_test, y_test, n_class)
        testloader = DataLoader(testset, batch_size, shuffle = True, num_workers = 0)

        model = module(trainset)
        model.to(device)

        train_accuracies, train_losses, val_accuracies, val_losses, stamped_out_dir = train(
            model = model, 
            trainloader = trainloader, 
            valloader = valloader, 
            epoch_num = epoch_num,
            record_freq = record_freq, 
            out_dir = out_dir)

        plotTrainingProcess(
            record_freq, 
            trainloader, 
            train_accuracies, 
            train_losses, 
            val_accuracies, 
            val_losses,
            stamped_out_dir
        )

        model.load_state_dict(torch.load(os.path.join(stamped_out_dir, "best_model.pth")))

        accuracy, loss, aucs = testPerformance(model, testloader, out_dir = stamped_out_dir)
        auc = np.average(aucs)

        kf_accuracies.append(accuracy)
        kf_losses.append(loss)
        kf_aucs.append(auc)

    # Close opened matplotlib figures
    plt.close("all")

    kf_accuracy = sum(kf_accuracies) / len(kf_accuracies)
    kf_loss = sum(kf_losses) / len(kf_losses)
    kf_auc = sum(kf_aucs) / len(kf_aucs)

    # Recording
    for i in range(n_splits):
        report = f"[fold {i + 1}] test accuracy: {kf_accuracies[i]}, test loss: {kf_losses[i]}, test AUC: {kf_aucs[i]}"
        log.log(report)

    report = f"[overall] test accuracy: {kf_accuracy}, test loss: {kf_loss}, test AUC: {kf_auc}"
    log.log(report)

    return kf_accuracy, kf_loss, kf_auc


def n_times_cross_validation_test(n_times: int, module: type, X: np.ndarray, y: np.ndarray, batch_size: int, epoch_num: int, out_dir: str = "temp"):
    """
    Run cross_validation_test ofr n times to get a more stable result.
    """

    # Create an output directory with a timestamp
    out_dir = os.path.join(out_dir, f"{module.__name__}_NCVT_{timestamp()}")
    while os.path.exists(out_dir):
        out_dir = out_dir + "_new"
    os.mkdir(out_dir)

    # Create log
    report_file_path = os.path.join(out_dir, "report.out")
    log = Log(report_file_path)

    # 
    kf_accuracies = []
    kf_losses = []
    kf_aucs = []

    for i in range(n_times):

        print(f"time {i}")

        kf_accuracy, kf_loss, kf_auc = cross_validation_test(
            module = CovidCNNModel,
            X = X,
            y = y,
            batch_size = batch_size,
            epoch_num = epoch_num,
            out_dir = out_dir
        )

        kf_accuracies.append(kf_accuracy)
        kf_losses.append(kf_loss)
        kf_aucs.append(kf_auc)

    accuracy = sum(kf_accuracies) / len(kf_accuracies)
    loss = sum(kf_losses) / len(kf_losses)
    auc = sum(kf_aucs) / len(kf_aucs)

    # Recording
    for i in range(n_times):
        report = f"[time {i + 1}] test accuracy: {kf_accuracies[i]}, test loss: {kf_losses[i]}, test AUC: {kf_aucs[i]}"
        log.log(report)

    report = f"[overall] test accuracy: {accuracy}, test loss: {loss}, test AUC: {auc}"
    log.log(report)


def timestamp():
    """
    Return the formated timestamp.
    """
    # Timestamp format
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    return timestamp