import numpy as np
import torch
import argparse
import math
import doasys

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_training', type=int, default=320, help='# of training data')
    parser.add_argument('--n_validation', type=int, default=64, help='# of validation data')
    parser.add_argument('--n_test', type=int, default=10, help='# of test data for various SNR')
    parser.add_argument('--grid_size', type=int, default=10000, help='the size of grids')
    parser.add_argument('--gaussian_std', type=float, default=100, help='the size of grids')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of batch')

    # Module parameters
    parser.add_argument('--n_layers', type=int, default=6, help='number of convolutional layers in the module')
    parser.add_argument('--n_filters', type=int, default=2, help='number of filters per layer in the module')
    parser.add_argument('--kernel_size', type=int, default=3, help='filter size in the convolutional blocks of the fr module')
    parser.add_argument('--inner_dim', type=int, default=32, help='dimension after first linear transformation')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate for adam optimizer used for the module')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs used to train the module')

    # Array parameters
    parser.add_argument('--ant_num', type=int, default=16, help='the number of antennas')
    parser.add_argument('--max_target_num', type=int, default=3, help='the maximum number of targets')
    parser.add_argument('--snr', type=float, default=1., help='the maximum SNR')
    parser.add_argument('--d', type=float, default=0.5, help='the distance between antennas')

    # Imperfect parameters
    parser.add_argument('--max_per_std', type=float, default=0.15, help='the maximum std of the position perturbation')
    parser.add_argument('--max_amp_std', type=float, default=0.5, help='the maximum std of the amplitude')
    parser.add_argument('--max_phase_std', type=float, default=0.2, help='the maximum std of the phase')
    parser.add_argument('--max_mc', type=float, default=0.06, help='the maximum mutual coupling (0.1->-10dB)')
    parser.add_argument('--nonlinear', type=float, default=1.0, help='the nonlinear parameter')
    parser.add_argument('--is_nonlinear', type=int, default=1, help='nonlinear effect')

    # Training policy
    parser.add_argument('--new_train', type=int, default=0, help='train a new network')
    parser.add_argument('--train_num', type=int, default=1, help='train a new network')
    parser.add_argument('--net_type', type=int, default=0, help='the type of network')

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    max_per_std = args.max_per_std
    max_amp_std = args.max_amp_std
    max_phase_std = args.max_phase_std
    max_mc = args.max_mc
    nonlinear = args.nonlinear
    is_nonlinear = args.is_nonlinear

    # Store test data for all SNR levels
    test_data_doa = []
    test_data_clean = []
    test_data_noisy = []
    test_data_num = []
    test_data_snr = []

    for idx in range(args.train_num):
        # Initialize lists to store data for all train_types
        training_data = {
            'doa': [],
            'clean_signal': [],
            'noisy_signal': [],
            'target_num': [],
            'train_type': []
        }

        validation_data = {
            'doa': [],
            'clean_signal': [],
            'noisy_signal': [],
            'target_num': [],
            'train_type': []
        }

        for train_type in range(7):
            if train_type == 0:
                args.max_per_std = 0
                args.max_amp_std = 0
                args.max_phase_std = 0
                args.max_mc = 0
                args.nonlinear = 0
                args.is_nonlinear = 0
            elif train_type == 1:
                args.max_per_std = max_per_std
                args.max_amp_std = 0
                args.max_phase_std = 0
                args.max_mc = 0
                args.nonlinear = 0
                args.is_nonlinear = 0
            elif train_type == 2:
                args.max_per_std = 0
                args.max_amp_std = max_amp_std
                args.max_phase_std = 0
                args.max_mc = 0
                args.nonlinear = 0
                args.is_nonlinear = 0
            elif train_type == 3:
                args.max_per_std = 0
                args.max_amp_std = 0
                args.max_phase_std = max_phase_std
                args.max_mc = 0
                args.nonlinear = 0
                args.is_nonlinear = 0
            elif train_type == 4:
                args.max_per_std = 0
                args.max_amp_std = 0
                args.max_phase_std = 0
                args.max_mc = max_mc
                args.nonlinear = 0
                args.is_nonlinear = 0
            elif train_type == 5:
                args.max_per_std = 0
                args.max_amp_std = 0
                args.max_phase_std = 0
                args.max_mc = 0
                args.nonlinear = nonlinear
                args.is_nonlinear = is_nonlinear
            else:
                args.max_per_std = max_per_std
                args.max_amp_std = max_amp_std
                args.max_phase_std = max_phase_std
                args.max_mc = max_mc
                args.nonlinear = nonlinear
                args.is_nonlinear = is_nonlinear

            # Generate the training data
            clean_signal, doa, target_num = doasys.gen_signal(args.n_training, args)

            clean_signal_tensor = torch.from_numpy(clean_signal).float()
            # print("shape of clean signal is: ", clean_signal_tensor.shape)
            # print("clean signal tensor: ",clean_signal_tensor)
            noisy_signal = doasys.noise_torch(clean_signal_tensor, args.snr)
            noisy_signal_np = noisy_signal.numpy()
            print("noisy signal is: ", noisy_signal)
            print("doa is: ", doa)
            print("target num is: ", target_num)

            # print("shape of noisy signal is: ", noisy_signal_np.shape)
            # print("noisy signal tensor: ",noisy_signal_np)

            # Append training data
            training_data['doa'].append(doa)
            training_data['clean_signal'].append(clean_signal)
            training_data['noisy_signal'].append(noisy_signal_np)
            training_data['target_num'].append(target_num)
            training_data['train_type'].append(train_type)

            # Generate the validation data
            clean_signal, doa, target_num = doasys.gen_signal(args.n_validation, args)
            clean_signal_tensor = torch.from_numpy(clean_signal).float()

            noisy_signal = doasys.noise_torch(clean_signal_tensor, args.snr)
            noisy_signal_np = noisy_signal.numpy()

            # Append validation data
            validation_data['doa'].append(doa)
            validation_data['clean_signal'].append(clean_signal)
            validation_data['noisy_signal'].append(noisy_signal_np)
            validation_data['target_num'].append(target_num)
            validation_data['train_type'].append(train_type)
        # print("train type: ", training_data['train_type'])        # Save all training data for this idx in one file
        # print("shape of dataset noisy signal is: ", training_data['noisy_signal'].size())
        print("data set doa is: ",training_data['doa'])
        np.savez(f'training_data_{idx}.npz',
                 doa=training_data['doa'],
                 clean_signal=training_data['clean_signal'],
                 noisy_signal=training_data['noisy_signal'],
                 target_num=training_data['target_num'],
                 train_type=training_data['train_type'],
                 d=args.d,
                 max_target_num=args.max_target_num,
                 ant_num=args.ant_num,
                 n_training=args.n_training
                 )

        # Save all validation data for this idx in one file
        np.savez(f'validation_data_{idx}.npz',
                 doa=validation_data['doa'],
                 clean_signal=validation_data['clean_signal'],
                 noisy_signal=validation_data['noisy_signal'],
                 target_num=validation_data['target_num'],
                 train_type=validation_data['train_type'],
                 d=args.d,
                 n_validation=args.n_validation
                 )

        # Generate test data only for idx == 0
        if idx == 0:
            # [Test data generation code remains the same]
            SNR_range = np.linspace(10, 30, 7)
            n_test = args.n_test

            for snr_idx, SNR_dB in enumerate(SNR_range):
                for n1 in range(n_test):
                    test_len = 2
                    clean_signal, doa, target_num = doasys.gen_signal(test_len, args)

                    clean_signal_tensor = torch.from_numpy(clean_signal).float()
                    noisy_signal = doasys.noise_torch(clean_signal_tensor, math.pow(10.0, SNR_dB / 10.0))
                    noisy_signal_np = noisy_signal.numpy()

                    test_data_doa.append(doa)
                    test_data_clean.append(clean_signal)
                    test_data_noisy.append(noisy_signal_np)
                    test_data_num.append(target_num)
                    test_data_snr.append(SNR_dB)

        # Save test data
    np.savez('test_data.npz',
             doa=test_data_doa,
             clean_signal=test_data_clean,
             noisy_signal=test_data_noisy,
             target_num=test_data_num,
             SNR = test_data_snr,
             d=args.d,
             n_test=args.n_test
             )

    print("Data generation completed successfully.")
    print(f"Generated {args.train_num} training and validation datasets.")