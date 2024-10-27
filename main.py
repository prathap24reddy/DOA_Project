import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.signal
import doasys
import matlab.engine
from scipy import io


def make_hankel(signal, m):
    """
    Auxiliary function used in MUSIC.
    """
    n = len(signal)
    h = np.zeros((m, n - m + 1), dtype='complex128')
    for r in range(m):
        for c in range(n - m + 1):
            h[r, c] = signal[r + c]
    return h


def music(signal, xgrid, nfreq, m=20):
    """
    Compute frequency representation obtained with MUSIC.
    """
    music_fr = np.zeros((signal.shape[0], len(xgrid)))
    for n in range(signal.shape[0]):
        hankel = make_hankel(signal[n], m)
        _, _, V = np.linalg.svd(hankel)
        v = np.exp(-2.0j * np.pi * np.outer(xgrid[:, None], np.arange(0, signal.shape[1] - m + 1)))
        u = V[nfreq[n]:]
        fr = -np.log(np.linalg.norm(np.tensordot(u, v, axes=(1, 1)), axis=0) ** 2)
        music_fr[n] = fr
    return music_fr


if __name__ == '__main__':
    # Configuration flags
    is_music = False
    is_anm = False
    is_proposed = True
    is_fig = False
    is_save = False

    # Load test data
    test_data = np.load('test_data.npz')
    doa = test_data['doa']
    noisy_signals = torch.from_numpy(test_data['noisy_signal']).float()
    target_num = test_data['target_num']
    SNR_range = test_data['SNR']
    d = float(test_data['d'])
    max_target_num = int(test_data['max_target_num'])
    ant_num = int(test_data['ant_num'])
    n_test = int(test_data['n_test'])

    print("noisy signal size: ", noisy_signals.shape)
    print("doa size: ", doa.shape)
    print("snr range size: ", SNR_range.shape)


    # Set device
    use_cuda = torch.cuda.is_available()

    # Grid setup
    grid_size = 10000
    doa_grid = np.linspace(-50, 50, grid_size, endpoint=False)
    ref_grid = doa_grid

    # Load pre-trained network
    if use_cuda:
        net = doasys.spectrumModule(signal_dim=ant_num, n_filters=2, inner_dim=32,
                                   n_layers=6, kernel_size=3)
        net.load_state_dict(torch.load('net.pth'))
        net = net.cuda()
    else:
        net = doasys.spectrumModule(signal_dim=ant_num, n_filters=2, inner_dim=32,
                                   n_layers=6, kernel_size=3)
        net.load_state_dict(torch.load('net.pth', map_location=torch.device('cpu')))

    # Set the network to evaluation mode
    net.eval()

    # Create dictionary matrix
    dic_mat = np.zeros((doa_grid.size, 2, ant_num))
    dic_mat_comp = np.zeros((doa_grid.size, ant_num), dtype=complex)
    for n in range(doa_grid.size):
        tmp = doasys.steer_vec(doa_grid[n], d, ant_num, np.zeros(ant_num).T)
        tmp = tmp / np.sqrt(np.sum(np.power(np.abs(tmp), 2)))
        dic_mat[n, 0] = tmp.real
        dic_mat[n, 1] = tmp.imag
        dic_mat_comp[n] = tmp
    dic_mat_torch = torch.from_numpy(dic_mat).float()
    if use_cuda:
        dic_mat_torch = dic_mat_torch.cuda()

    # Initialize RMSE arrays
    RMSE = np.zeros((SNR_range.size, 1))
    RMSE_FFT = np.zeros((SNR_range.size, 1))
    RMSE_MUSIC = np.zeros((SNR_range.size, 1))
    RMSE_OMP = np.zeros((SNR_range.size, 1))
    RMSE_ANM = np.zeros((SNR_range.size, 1))

    # Start MATLAB engine for MUSIC and ANM
    eng = matlab.engine.start_matlab()

    # Process each SNR level
    for n in range(SNR_range.size):
        RMSE[n] = RMSE_FFT[n] = RMSE_MUSIC[n] = RMSE_OMP[n] = RMSE_ANM[n] = 0

        for n1 in range(n_test):
            epoch_start_time = time.time()

            # Get current batch of signals
            current_signals = noisy_signals[n*n_test+n1:n*n_test+n1 + 1]  # Using single sample for demonstration
            current_doa = doa[n*n_test+n1:n*n_test+n1 + 1]
            current_target_num = target_num[n*n_test+n1:n*n_test+n1 + 1]

            current_signals = current_signals.squeeze()  # Shape: [2, 2, 16]
            current_doa = current_doa.squeeze()  # Shape: (2, 3)
            current_target_num = current_target_num.squeeze()  # Shape: (2,)
            # print("n is: ", n, "target num is: ", current_target_num)
            print("current signal shape is: ", current_signals.shape)
            print("current doa shape is: ", current_doa.shape)
            print("current target num shape is: ", current_target_num.shape)
            # Proposed method
            if is_proposed:
                if use_cuda:
                    current_signals = current_signals.cuda()
                with torch.no_grad():
                    output_net = net(current_signals).view(2, 2, -1)

                mm_real = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 0, :].T) + torch.mm(output_net[:, 1, :],
                                                                                             dic_mat_torch[:, 1, :].T)
                mm_imag = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 1, :].T) - torch.mm(output_net[:, 1, :],
                                                                                             dic_mat_torch[:, 0, :].T)
                sp = torch.pow(mm_real, 2) + torch.pow(mm_imag, 2)
                sp_np = sp.cpu().detach().numpy()

                for idx_sp in range(sp_np.shape[0]):
                    sp_np[idx_sp] = sp_np[idx_sp] / np.max(sp_np[idx_sp])

                doa_num = (current_doa >= -90).sum(axis=1)
                est_doa = doasys.get_doa(sp_np, doa_num, doa_grid, max_target_num, current_doa)
                RMSE[n] += np.sum(np.power(np.abs(est_doa - current_doa), 2))
                print("doa is: ", current_doa)
                print("estimated doa is: ", est_doa)

            # FFT method
            r = current_signals.cpu().detach().numpy() if use_cuda else current_signals.detach().numpy()
            r_c = r[:, 0, :] + 1j * r[:, 1, :]
            sp_FFT = np.power(np.abs(np.matmul(dic_mat_comp, np.conj(r_c).T)), 2).T

            for idx_sp in range(sp_FFT.shape[0]):
                sp_FFT[idx_sp] = sp_FFT[idx_sp] / np.max(sp_FFT[idx_sp])

            doa_num = (current_doa >= -90).sum(axis=1)
            est_doa = doasys.get_doa(sp_FFT, doa_num, doa_grid, max_target_num, current_doa)
            RMSE_FFT[n] += np.sum(np.power(np.abs(est_doa - current_doa), 2))

            # MUSIC algorithm
            if is_music:
                r_c = r[:, 0, :] + 1j * r[:, 1, :]
                sp_MUSIC = np.zeros((r_c.shape[0], grid_size))
                for idx_r in range(r_c.shape[0]):
                    x_tmp = eng.MUSIConesnapshot(matlab.double(list(r_c[idx_r]), is_complex=True),
                                                 int(current_target_num[idx_r]),
                                                 matlab.double(list(doa_grid), is_complex=False))
                    sp_MUSIC[idx_r] = np.squeeze(np.asarray(x_tmp))

                est_doa = doasys.get_doa(sp_MUSIC, doa_num, doa_grid, max_target_num, current_doa)
                RMSE_MUSIC[n] += np.sum(np.power(np.abs(est_doa - current_doa), 2))

            # OMP algorithm
            est_doa_omp = -100 * np.ones((r_c.shape[0], max_target_num))
            for idx1 in range(r_c.shape[0]):
                r_tmp0 = np.expand_dims(r_c[idx1], axis=0)
                r_tmp1 = r_tmp0
                max_idx = np.zeros(current_target_num[idx1], dtype=int)
                for idx2 in range(current_target_num[idx1]):
                    max_idx_tmp = np.argmax(np.abs(np.matmul(dic_mat_comp, np.conj(r_tmp1).T)))
                    max_idx[idx2] = max_idx_tmp
                    dic_tmp = dic_mat_comp[max_idx[0:idx2 + 1]]
                    r_tmp1 = r_tmp0 - np.matmul(np.matmul(r_tmp0, np.linalg.pinv(dic_tmp)), dic_tmp)
                    est_doa_omp[idx1, idx2] = doa_grid[max_idx_tmp]
                est_doa_omp[idx1] = np.sort(est_doa_omp[idx1])
            RMSE_OMP[n] += np.sum(np.power(np.abs(est_doa_omp - current_doa), 2))

            # ANM algorithm
            if is_anm:
                x = np.zeros((r_c.shape[0], ant_num), dtype=complex)
                for idx_r in range(r_c.shape[0]):
                    x_tmp = eng.ANM(matlab.double(list(r_c[idx_r]), is_complex=True))
                    x[idx_r] = np.squeeze(np.asarray(x_tmp))

                sp_ANM = np.power(np.abs(np.matmul(dic_mat_comp, np.conj(x).T)), 2).T
                for idx_sp in range(sp_ANM.shape[0]):
                    sp_ANM[idx_sp] = sp_ANM[idx_sp] / np.max(sp_ANM[idx_sp])

                est_doa = doasys.get_doa(sp_ANM, doa_num, doa_grid, max_target_num, current_doa)
                RMSE_ANM[n] += np.sum(np.power(np.abs(est_doa - current_doa), 2))

            # Plotting
            if is_fig:
                plt.figure()
                if is_proposed:
                    plt.plot(doa_grid, sp_np[0], label='Proposed method')
                plt.plot(doa_grid, sp_FFT[0], label='FFT method')
                if is_anm:
                    plt.plot(doa_grid, sp_ANM[0], label='ANM method')
                if is_music:
                    plt.plot(doa_grid, sp_MUSIC[0], label='MUSIC method')

                tmp_doa = est_doa_omp[0][np.argwhere(est_doa_omp[0] > -90)]
                plt.stem(tmp_doa, np.ones((tmp_doa.size, 1)), label='OMP method')
                tmp_doa = current_doa[0][np.argwhere(current_doa[0] > -90)]
                plt.stem(tmp_doa, np.ones((tmp_doa.size, 1)), label='Ground-truth DOA')
                plt.xlabel('Spatial angle (deg)')
                plt.ylabel('Spatial spectrum')
                plt.legend()
                plt.grid()
                plt.show()

            print(f"SNR: {SNR_range[n]:.2f} dB, Test: {n1}/{n_test}, Time: {time.time() - epoch_start_time:.2f}")

        # Calculate final RMSE values
        RMSE[n] = np.sqrt(RMSE[n] / (current_doa.size * n_test))
        RMSE_FFT[n] = np.sqrt(RMSE_FFT[n] / (current_doa.size * n_test))
        if is_music:
            RMSE_MUSIC[n] = np.sqrt(RMSE_MUSIC[n] / (current_doa.size * n_test))
        RMSE_OMP[n] = np.sqrt(RMSE_OMP[n] / (current_doa.size * n_test))
        if is_anm:
            RMSE_ANM[n] = np.sqrt(RMSE_ANM[n] / (current_doa.size * n_test))

        print(f"SNR (dB): {SNR_range[n]:.2f}, RMSE (deg): {float(RMSE[n]):.2f}, "
              f"RMSE_FFT (deg): {float(RMSE_FFT[n]):.2f}, RMSE_OMP (deg): {float(RMSE_OMP[n]):.2f}, "
              f"RMSE_ANM (deg): {float(RMSE_ANM[n]):.2f}, RMSE_MUSIC (deg): {float(RMSE_MUSIC[n]):.2f}")

    # Final plotting
    plt.figure()
    plt.semilogy(SNR_range, RMSE, linestyle='-', marker='o', linewidth=2, markersize=8, label='Proposed method')
    plt.semilogy(SNR_range, RMSE_FFT, linestyle='-', marker='v', linewidth=2, markersize=8, label='FFT method')
    plt.semilogy(SNR_range, RMSE_MUSIC, linestyle='-', marker='x', linewidth=2, markersize=8, label='MUSIC method')
    plt.semilogy(SNR_range, RMSE_OMP, linestyle='-', marker='+', linewidth=2, markersize=8, label='OMP method')
    plt.semilogy(SNR_range, RMSE_ANM, linestyle='-', marker='s', linewidth=2, markersize=8, label='ANM method')
    plt.xlabel('SNR (dB)')
    plt.ylabel('RMSE (deg)')
    plt.legend()
    plt.grid()
    plt.show()

    # Save results if requested
    if is_save:
        io.savemat('SNR_range.mat', {'array': SNR_range})
        io.savemat('RMSE.mat', {'array': RMSE})
        io.savemat('RMSE_FFT.mat', {'array': RMSE_FFT})
        io.savemat('RMSE_MUSIC.mat', {'array': RMSE_MUSIC})
        io.savemat('RMSE_OMP.mat', {'array': RMSE_OMP})
        io.savemat('RMSE_ANM.mat', {'array': RMSE_ANM})