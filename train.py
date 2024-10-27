import numpy as np
import torch
import argparse
import torch.utils.data as data_utils
import doasys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs used to train the module')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of batch')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate for Adam optimizer')
    parser.add_argument('--grid_size', type=int, default=10000, help='the size of grids')
    parser.add_argument('--gaussian_std', type=float, default=100, help='the size of Gaussian distribution for DOA')

    # Array parameters
    parser.add_argument('--n_training', type=int, default=320, help='# of training data')
    parser.add_argument('--n_validation', type=int, default=64, help='# of validation data')
    parser.add_argument('--n_test', type=int, default=10, help='# of test data for various SNR')
    parser.add_argument('--ant_num', type=int, default=16, help='the number of antennas')
    parser.add_argument('--d', type=float, default=0.5, help='the distance between antennas')
    parser.add_argument('--max_target_num', type=int, default=3, help='the maximum number of targets')

    # Network parameters
    parser.add_argument('--n_layers', type=int, default=6, help='number of convolutional layers in the module')
    parser.add_argument('--n_filters', type=int, default=2, help='number of filters per layer in the module')
    parser.add_argument('--inner_dim', type=int, default=32, help='dimension after first linear transformation')
    parser.add_argument('--kernel_size', type=int, default=3, help='kernel size in the convolutional blocks')

    # Load saved training and validation data
    parser.add_argument('--train_data_file', type=str, default='training_data_0.npz', help='path to training data file')
    parser.add_argument('--val_data_file', type=str, default='validation_data_0.npz',
                        help='path to validation data file')

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    # Load saved training data
    train_data = np.load(args.train_data_file)
    train_doa = train_data['doa']
    train_noisy_signal = train_data['noisy_signal']
    args.d = train_data['d']
    args.max_target_num = train_data['max_target_num']
    args.ant_num = train_data['ant_num']
    args.n_training=train_data['n_training']


    # Reshape doa to shape (_, max_target_num)
    train_doa = train_doa.reshape(-1, args.max_target_num)

    # Reshape noisy signal to shape (_, 2, ant_num)
    train_noisy_signal = train_noisy_signal.reshape(-1, 2, args.ant_num)

    # Load saved validation data
    val_data = np.load(args.val_data_file)
    val_doa = val_data['doa']
    val_noisy_signal = val_data['noisy_signal']
    args.n_validation=val_data['n_validation']

    # Reshape validation data
    val_doa = val_doa.reshape(-1, args.max_target_num)
    val_noisy_signal = val_noisy_signal.reshape(-1, 2, args.ant_num)

    # Generate the reference spectrum for training data
    doa_grid = np.linspace(-50, 50, args.grid_size, endpoint=False)
    ref_grid = doa_grid
    train_ref_sp = np.array(doasys.gen_refsp(train_doa, ref_grid, args.gaussian_std / args.ant_num))

    # Generate the reference spectrum for validation data
    val_ref_sp = np.array(doasys.gen_refsp(val_doa, ref_grid, args.gaussian_std / args.ant_num))

    # Convert data to tensors
    train_noisy_signal_tensor = torch.from_numpy(train_noisy_signal).float()
    train_ref_sp_tensor = torch.from_numpy(train_ref_sp).float()
    train_doa_tensor = torch.from_numpy(train_doa).float()
    # print("train noisy signal is: ", train_noisy_signal_tensor)
    # print("doa is: ", train_doa)
    val_noisy_signal_tensor = torch.from_numpy(val_noisy_signal).float()
    val_ref_sp_tensor = torch.from_numpy(val_ref_sp).float()
    val_doa_tensor = torch.from_numpy(val_doa).float()

    # Create TensorDatasets
    train_dataset = data_utils.TensorDataset(train_noisy_signal_tensor, train_ref_sp_tensor, train_doa_tensor)
    val_dataset = data_utils.TensorDataset(val_noisy_signal_tensor, val_ref_sp_tensor, val_doa_tensor)

    # Create DataLoaders
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the network
    net = doasys.spectrumModule(signal_dim=args.ant_num, n_filters=args.n_filters, inner_dim=args.inner_dim,
                                n_layers=args.n_layers, kernel_size=args.kernel_size)
    if args.use_cuda:
        net.cuda()

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss(reduction='sum')

    # Train the network using the train_net function
    for epoch in range(args.n_epochs):
        net, train_loss, val_loss = doasys.train_net(args, net, optimizer, criterion, train_loader, val_loader,
                                                     doa_grid, epoch, train_num=1, train_type=0, net_type=0)
        print(f'Epoch [{epoch + 1}/{args.n_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Save the model
    torch.save(net.state_dict(), 'net.pth')
    print("Training complete and model saved.")
