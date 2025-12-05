"""
Training from the paper, available at the following repo: https://github.com/ptitbroussou/Photonic_Subspace_QML_Toolkit
"""

import time

import torch
import torch.nn.functional as F  # noqa: N812


#####################################################################################################################
### Density Matrix Simulations                                                                                    ###
#####################################################################################################################
def to_density_matrix(batch_vectors, device):
    out = torch.zeros(
        [batch_vectors.size(0), batch_vectors.size(1), batch_vectors.size(1)]
    ).to(device)
    index = 0
    for vector in batch_vectors:
        out[index] += torch.einsum("i,j->ij", vector, vector)
        index += 1
    return out


def normalize_dm(density_matrix):
    traces = density_matrix.diagonal(dim1=-2, dim2=-1).sum(-1)
    traces = traces.view(density_matrix.shape[0], 1, 1)
    return density_matrix / traces


def train_network_2d(
    batch_size, d, network, train_loader, criterion, output_scale, optimizer, device
):
    network.train()  # put in train mode: we will modify the weights of the network
    train_loss = 0  # initialize the loss
    train_accuracy = 0  # initialize the accuracy

    # loop on the batches in the train dataset
    for _batch_idx, (data, target) in enumerate(train_loader):
        temp_batch_size = data.size(0)
        optimizer.zero_grad()  # important step to reset gradients to zero
        data = data.to(device)
        init_density_matrix = to_density_matrix(
            F.normalize(data.squeeze().resize(data.shape[0], d**2), p=2, dim=1).to(
                device
            ),
            device,
        )
        output = network(
            normalize_dm(init_density_matrix)
        )  # we run the network on the data

        # training
        # print(output)
        # print(target)
        loss = criterion(
            output * output_scale, target.resize(temp_batch_size).to(device)
        )  # we compare output to the target and compute the loss, using the chosen loss function
        train_loss += loss.item()  # we increment the total train loss
        loss.backward()
        optimizer.step()

        # predict
        pred = output.argmax(dim=1, keepdim=True).to(
            device
        )  # the class chosen by the network is the highest output
        acc = (
            pred.eq(target.to(device).view_as(pred)).sum().item()
        )  # the accuracy is the proportion of correct classes
        train_accuracy += acc  # increment accuracy of whole test set

    train_accuracy /= len(train_loader.dataset)  # compute mean accuracy
    train_loss /= _batch_idx + 1  # mean loss
    return train_loss, train_accuracy


def test_network_2d(
    batch_size, d, network, test_loader, criterion, output_scale, device
):
    network.eval()  # put in eval mode: we will not modify the weights of the network
    train_loss = 0  # initialize the loss
    train_accuracy = 0  # initialize the accuracy
    for _batch_idx, (data, target) in enumerate(test_loader):
        temp_batch_size = data.size(0)
        # Run the network and compute the loss
        data = data.to(device)
        init_density_matrix = to_density_matrix(
            F.normalize(data.squeeze().resize(data.shape[0], d**2), p=2, dim=1).to(
                device
            ),
            device,
        )
        output = network(
            normalize_dm(init_density_matrix)
        )  # we run the network on the data
        loss = criterion(
            output * output_scale, target.resize(temp_batch_size).to(device)
        )  # we compare output to the target and compute the loss, using the chosen loss function
        train_loss += loss.item()  # we increment the total train loss
        pred = output.argmax(
            dim=1, keepdim=True
        )  # the class chosen by the network is the highest output
        acc = (
            pred.eq(target.to(device).view_as(pred)).sum().item()
        )  # the accuracy is the proportion of correct classes
        train_accuracy += acc  # increment accuracy of whole test set

    train_accuracy /= len(test_loader.dataset)  # compute mean accuracy
    train_loss /= _batch_idx + 1  # mean loss
    return train_loss, train_accuracy


def train_globally(
    batch_size,
    d,
    network,
    reduced_train_loader,
    reduced_test_loader,
    optimizer,
    scheduler,
    criterion,
    output_scale,
    train_epochs,
    test_interval,
    device,
):
    """
    Perform general training on the single channel image and single channel network, including training, testing, and saving data.
    """
    # print number of parameters of network
    total_params = sum(p.numel() for p in network.parameters())
    print(f"Start training! Number of network total parameters: {total_params}")

    test_loss, test_accuracy = test_network_2d(
        batch_size, d, network, reduced_test_loader, criterion, output_scale, device
    )
    print(
        f"Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %"
    )

    # training step
    training_loss_list = []
    testing_loss_list = [test_loss]
    training_accuracy_list = []
    testing_accuracy_list = [test_accuracy]
    for epoch in range(train_epochs):
        start = time.time()
        train_loss, train_accuracy = train_network_2d(
            batch_size,
            d,
            network,
            reduced_train_loader,
            criterion,
            output_scale,
            optimizer,
            device,
        )
        training_loss_list.append(train_loss)
        training_accuracy_list.append(train_accuracy)
        end = time.time()
        print(
            f"Epoch {epoch}: Loss = {train_loss:.6f}, accuracy = {train_accuracy * 100:.4f} %, time={(end - start):.4f}s"
        )
        if epoch % test_interval == 0 and epoch != 0:
            test_loss, test_accuracy = test_network_2d(
                batch_size,
                d,
                network,
                reduced_test_loader,
                criterion,
                output_scale,
                device,
            )
            testing_loss_list.append(test_loss)
            testing_accuracy_list.append(test_accuracy)
            print(
                f"Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %"
            )
        scheduler.step()
    # final testing part
    # test_loss, test_accuracy = test_network_2D(batch_size, d, network, reduced_test_loader, criterion, output_scale, device)
    # print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')
    return (
        network.state_dict(),
        training_loss_list,
        training_accuracy_list,
        testing_loss_list,
        testing_accuracy_list,
    )


#####################################################################################################################
### State Vector Simulations                                                                                      ###
#####################################################################################################################
def train_network_2d_state_vector(
    batch_size, d, network, train_loader, criterion, output_scale, optimizer, device
):
    network.train()  # put in train mode: we will modify the weights of the network
    train_loss = 0  # initialize the loss
    train_accuracy = 0  # initialize the accuracy

    # loop on the batches in the train dataset
    for _batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # important step to reset gradients to zero
        data = data.to(device)
        init_state_vector = F.normalize(
            data.squeeze().resize(data.shape[0], d**2), p=2, dim=1
        ).to(device)
        output = network(init_state_vector)  # we run the network on the data
        # training
        loss = criterion(
            output * output_scale, target.resize(batch_size).to(device)
        )  # we compare output to the target and compute the loss, using the chosen loss function
        train_loss += loss.item()  # we increment the total train loss
        loss.backward()
        optimizer.step()

        # predict
        pred = output.argmax(dim=1, keepdim=True).to(
            device
        )  # the class chosen by the network is the highest output
        acc = (
            pred.eq(target.to(device).view_as(pred)).sum().item()
        )  # the accuracy is the proportion of correct classes
        train_accuracy += acc  # increment accuracy of whole test set

    train_accuracy /= len(train_loader.dataset)  # compute mean accuracy
    train_loss /= _batch_idx + 1  # mean loss
    return train_loss, train_accuracy


def test_network_2d_state_vector(
    batch_size, d, network, test_loader, criterion, output_scale, device
):
    network.eval()  # put in eval mode: we will not modify the weights of the network
    train_loss = 0  # initialize the loss
    train_accuracy = 0  # initialize the accuracy
    for _batch_idx, (data, target) in enumerate(test_loader):
        # Run the network and compute the loss
        data = data.to(device)
        init_state_vector = F.normalize(
            data.squeeze().resize(data.shape[0], d**2), p=2, dim=1
        ).to(device)
        print(init_state_vector.shape)
        print(init_state_vector)
        print(torch.norm(init_state_vector[0]))
        output = network(init_state_vector)  # we run the network on the data
        loss = criterion(
            output * output_scale, target.resize(batch_size).to(device)
        )  # we compare output to the target and compute the loss, using the chosen loss function
        train_loss += loss.item()  # we increment the total train loss
        pred = output.argmax(
            dim=1, keepdim=True
        )  # the class chosen by the network is the highest output
        acc = (
            pred.eq(target.to(device).view_as(pred)).sum().item()
        )  # the accuracy is the proportion of correct classes
        train_accuracy += acc  # increment accuracy of whole test set

    train_accuracy /= len(test_loader.dataset)  # compute mean accuracy
    train_loss /= _batch_idx + 1  # mean loss
    return train_loss, train_accuracy


def train_globally_state_vector(
    batch_size,
    d,
    network,
    reduced_train_loader,
    reduced_test_loader,
    optimizer,
    scheduler,
    criterion,
    output_scale,
    train_epochs,
    test_interval,
    device,
):
    """
    Perform general training on the single channel image and single channel network, including training, testing, and saving data.
    """
    # print number of parameters of network
    total_params = sum(p.numel() for p in network.parameters())
    print(f"Start training! Number of network total parameters: {total_params}")

    test_loss, test_accuracy = test_network_2d_state_vector(
        batch_size, d, network, reduced_test_loader, criterion, output_scale, device
    )
    print(
        f"Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %"
    )

    # training step
    training_loss_list = []
    testing_loss_list = [test_loss]
    training_accuracy_list = []
    testing_accuracy_list = [test_accuracy]
    for epoch in range(train_epochs):
        start = time.time()
        train_loss, train_accuracy = train_network_2d_state_vector(
            batch_size,
            d,
            network,
            reduced_train_loader,
            criterion,
            output_scale,
            optimizer,
            device,
        )
        training_loss_list.append(train_loss)
        training_accuracy_list.append(train_accuracy)
        end = time.time()
        print(
            f"Epoch {epoch}: Loss = {train_loss:.6f}, accuracy = {train_accuracy * 100:.4f} %, time={(end - start):.4f}s"
        )
        if epoch % test_interval == 0 and epoch != 0:
            test_loss, test_accuracy = test_network_2d_state_vector(
                batch_size,
                d,
                network,
                reduced_test_loader,
                criterion,
                output_scale,
                device,
            )
            testing_loss_list.append(test_loss)
            testing_accuracy_list.append(test_accuracy)
            print(
                f"Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %"
            )
        scheduler.step()
    # final testing part
    # test_loss, test_accuracy = test_network_2D(batch_size, d, network, reduced_test_loader, criterion, output_scale, device)
    # print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')
    return (
        network.state_dict(),
        training_loss_list,
        training_accuracy_list,
        testing_loss_list,
        testing_accuracy_list,
    )


#####################################################################################################################
### MSE Neural Network                                                                                            ###
#####################################################################################################################
def train_network_2d_mse(
    batch_size, d, network, train_loader, criterion, output_scale, optimizer, device
):
    network.train()  # put in train mode: we will modify the weights of the network
    train_loss = 0  # initialize the loss
    train_accuracy = 0  # initialize the accuracy

    # loop on the batches in the train dataset
    for _batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # important step to reset gradients to zero
        data = data.to(device)
        init_density_matrix = to_density_matrix(
            F.normalize(data.squeeze().resize(data.shape[0], d**2), p=2, dim=1).to(
                device
            ),
            device,
        )
        output = network(
            normalize_dm(init_density_matrix)
        )  # we run the network on the data

        # training
        # print(output)
        # print(target)
        loss = criterion(
            output * output_scale, target.to(dtype=torch.float, device=device)
        )  # we compare output to the target and compute the loss, using the chosen loss function
        train_loss += loss.item()  # we increment the total train loss
        loss.backward()
        optimizer.step()

        # predict
        pred = output.argmax(
            dim=1, keepdim=True
        )  # the class chosen by the network is the highest output
        # acc = pred.eq(target.argmax(dim=1)).sum().item()  # the accuracy is the proportion of correct classes
        acc = pred.eq(target.argmax(dim=1).view(-1, 1)).sum().item()
        train_accuracy += acc  # increment accuracy of whole test set

    train_accuracy /= len(train_loader.dataset)  # compute mean accuracy
    train_loss /= _batch_idx + 1  # mean loss
    return train_loss, train_accuracy


def test_network_2d_mse(
    batch_size, d, network, test_loader, criterion, output_scale, device
):
    network.eval()  # put in eval mode: we will not modify the weights of the network
    train_loss = 0  # initialize the loss
    train_accuracy = 0  # initialize the accuracy
    for _batch_idx, (data, target) in enumerate(test_loader):
        # Run the network and compute the loss
        data = data.to(device)
        init_density_matrix = to_density_matrix(
            F.normalize(data.squeeze().resize(data.shape[0], d**2), p=2, dim=1).to(
                device
            ),
            device,
        )
        output = network(
            normalize_dm(init_density_matrix)
        )  # we run the network on the data
        loss = criterion(
            output * output_scale, target.to(dtype=torch.float, device=device)
        )  # we compare output to the target and compute the loss, using the chosen loss function
        train_loss += loss.item()  # we increment the total train loss
        pred = output.argmax(
            dim=1, keepdim=True
        )  # the class chosen by the network is the highest output
        # acc = pred.eq(target.argmax(dim=1)).sum().item()  # the accuracy is the proportion of correct classes
        acc = pred.eq(target.argmax(dim=1).view(-1, 1)).sum().item()
        train_accuracy += acc  # increment accuracy of whole test set

    train_accuracy /= len(test_loader.dataset)  # compute mean accuracy
    train_loss /= _batch_idx + 1  # mean loss
    return train_loss, train_accuracy


def train_globally_mse(
    batch_size,
    d,
    network,
    reduced_train_loader,
    reduced_test_loader,
    optimizer,
    scheduler,
    criterion,
    output_scale,
    train_epochs,
    test_interval,
    device,
):
    """
    Perform general training on the single channel image and single channel network, including training, testing, and saving data.
    """
    # print number of parameters of network
    total_params = sum(p.numel() for p in network.parameters())
    print(f"Start training! Number of network total parameters: {total_params}")

    test_loss, test_accuracy = test_network_2d_mse(
        batch_size, d, network, reduced_test_loader, criterion, output_scale, device
    )
    print(
        f"Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %"
    )

    # training step
    training_loss_list = []
    testing_loss_list = [test_loss]
    training_accuracy_list = []
    testing_accuracy_list = [test_accuracy]
    for epoch in range(train_epochs):
        start = time.time()
        train_loss, train_accuracy = train_network_2d_mse(
            batch_size,
            d,
            network,
            reduced_train_loader,
            criterion,
            output_scale,
            optimizer,
            device,
        )
        training_loss_list.append(train_loss)
        training_accuracy_list.append(train_accuracy)
        end = time.time()
        print(
            f"Epoch {epoch}: Loss = {train_loss:.6f}, accuracy = {train_accuracy * 100:.4f} %, time={(end - start):.4f}s"
        )
        if epoch % test_interval == 0 and epoch != 0:
            test_loss, test_accuracy = test_network_2d_mse(
                batch_size,
                d,
                network,
                reduced_test_loader,
                criterion,
                output_scale,
                device,
            )
            testing_loss_list.append(test_loss)
            testing_accuracy_list.append(test_accuracy)
            print(
                f"Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %"
            )
        scheduler.step()
    # final testing part
    # test_loss, test_accuracy = test_network_2D(batch_size, d, network, reduced_test_loader, criterion, output_scale, device)
    # print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')
    return (
        network.state_dict(),
        training_loss_list,
        training_accuracy_list,
        testing_loss_list,
        testing_accuracy_list,
    )
