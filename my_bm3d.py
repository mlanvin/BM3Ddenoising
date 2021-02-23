from itertools import product
import numpy as np
import pywt
import tqdm
from scipy.fft import fftn, ifftn


class Group3d:
    def __init__(self, i_R, j_R, group_3d, bloc_coord, N_size):
        self.i_R = i_R
        self.j_R = j_R
        self.group_3d = group_3d
        self.bloc_coord = bloc_coord  # first one is (i_R, j_R)
        self.N_size = N_size

    def bloc(self, im, jm):
        k = np.where(self.bloc_coord == (im, jm))
        return self.group_3d[:, :, k]

    def is_bloc(self, im, jm):
        return np.where(self.bloc_coord == (im, jm)) != []

    def all_values(self, i, j):
        values = []
        N = self.N_size
        for k, (ii, jj) in enumerate(self.bloc_coord):
            if k >= self.group_3d.shape[0]:
                continue
            if ii <= i < ii + N and jj <= j < jj + N:
                # print(self.group_3d.shape, len(self.bloc_coord))
                values.append(self.group_3d[k, i - ii, j - jj])
        return values


class BM3D:
    def __init__(self, noisy_img, **kwargs):
        self.N = noisy_img.shape[0]
        self.N1_th = kwargs.get("N1_th")
        self.N1_wie = kwargs.get("N1_wie")
        self.Ns = kwargs.get("Ns")
        self.N_step = kwargs.get("N_step")
        self.sigma = kwargs.get("sigma")
        self.lambda_3d = kwargs.get("lambda_3d")
        self.lambda_2d = kwargs.get("lambda_2d")
        self.tau_ht_match = kwargs.get("tau_ht_match")
        self.tau_wie_match = kwargs.get("tau_wie_match")

        self.w_th = np.zeros((self.N, self.N))
        self.w_wie = np.zeros((self.N, self.N))

        self.th_itf_3d = np.empty((self.N, self.N), dtype=object)  # list of Group3d
        self.wie_itf_3d = np.empty((self.N, self.N), dtype=object)  # list of Group3d
        self.wiener_energies = np.zeros((self.N, self.N), dtype=object)

        # padding :
        self.pad_w = max(self.N1_wie, self.N1_th)
        N_ = self.N + self.pad_w
        self.img = np.zeros((N_, N_))
        self.img[:noisy_img.shape[0], :noisy_img.shape[1]] = noisy_img

        self.img_basic_estimate = np.zeros((N_, N_))
        self.img_final_estimate = np.zeros((self.N, self.N))

        self.stat_bloc_similar = np.zeros((self.N, self.N))

    def denoise(self):
        """
            Denoise self.img according to the algorithm described in the paper
        :return: 2d np array, same size as the input image
        """

        # Step 1 : Basic Estimate
        for i, j in product(range(self.N), repeat=2):
            group_x_R_th, bloc_coord = self.grouping_from_noisy(i, j)

            tf_3d = self.transformation_3d(group_x_R_th)
            thresholded, N_xR_har = self.hard_threshold(tf_3d)

            self.w_th[i, j] = self.weight_th(N_xR_har)

            itf_3d = self.itransformation_3d(thresholded)
            self.th_itf_3d[i, j] = Group3d(i, j, itf_3d, bloc_coord, self.N1_th)

        print(self.stat_bloc_similar)
        self.compute_y_basic()

        # Step 2 : Final Estimate
        for i, j in product(range(self.N), repeat=2):
            group_xR_noisy, _ = self.grouping_from_noisy(i, j)
            group_xR_basic, bloc_coord_basic = self.grouping_from_basic_estimate(i, j)

            tf_3d_noisy = self.transformation_3d(group_xR_noisy)
            tf_3d_basic = self.transformation_3d(group_xR_basic)

            self.compute_wiener_energy(i, j, tf_3d_basic)
            self.w_wie[i, j] = self.weight_wie(i, j)

            wienered = self.wiener_filter(tf_3d_noisy, i, j)
            self.wie_itf_3d[i, j] = Group3d(i, j, self.itransformation_3d(wienered), bloc_coord_basic, self.N1_wie)

        self.compute_y_final()

        return self.img_final_estimate[:self.N - self.pad_w, :self.N - self.pad_w]

    def grouping_from_noisy(self, i, j):
        # use self.img
        N1 = self.N1_th
        N_step = self.N_step
        Ns = self.Ns
        tau = self.tau_ht_match
        return self._grouping(i, j, N1, N_step, Ns, self.img, tau)

    def grouping_from_basic_estimate(self, i, j):
        # use self.img_basic_estimate
        N1 = self.N1_wie
        N_step = self.N_step
        Ns = self.Ns
        tau = self.tau_wie_match
        return self._grouping(i, j, N1, N_step, Ns, self.img_basic_estimate, tau)

    def _grouping(self, i, j, N1, N_step, Ns, img, tau):
        S_xR = []
        bloc_coord = []
        delta_x = (max(0, i - Ns), min(self.N, i + Ns))
        delta_y = (max(0, j - Ns), min(self.N, j + Ns))
        this_bloc = img[i:i + N1, j:j + N1]

        for ii in range(delta_x[0], delta_x[1], N_step):
            for jj in range(delta_y[0], delta_y[1], N_step):
                if ii + N1 >= delta_x[1] or jj + N1 >= delta_y[1]:
                    continue
                bloc = img[ii:ii + N1, jj:jj + N1]
                if self.bloc_similarity(bloc, this_bloc, N1) < tau:
                    S_xR.append(bloc)
                    bloc_coord.append((ii, jj))

        if (i, j) not in bloc_coord:
            S_xR.append(this_bloc)
            bloc_coord.append((i, j))

        self.stat_bloc_similar[i, j] = len(bloc_coord)

        return S_xR, bloc_coord

    @staticmethod
    def bloc_similarity(b1, b2, N):
        norm = np.linalg.norm(b1 - b2)
        val = (norm / N) ** 2
        return val

    @staticmethod
    def transformation_3d(group):
        # return pywt.dwtn(group, 'bior1.5')
        return fftn(group)

    @staticmethod
    def itransformation_3d(group):
        # return pywt.idwtn(group, 'bior1.5')
        return ifftn(group)

    def hard_threshold_direction(self, tf_3d_direction):
        idx = tf_3d_direction < self.lambda_3d
        thresh = np.zeros(tf_3d_direction.shape)
        thresh[idx] = tf_3d_direction[idx]
        N_retained_values_direction = np.sum(idx)
        return thresh, N_retained_values_direction

    def hard_threshold(self, tf_3d):
        """Perform the hard thresholding

        Args:
            tf_3d ([array]): [array to threshold]
        """
        N_retained_values = 0
        tf_3d, N_retained_values_direction = self.hard_threshold_direction(tf_3d)
        # for key, tf_3d_direction in tf_3d.items():
        #     tf_3d[key], N_retained_values_direction = self.hard_threshold_direction(tf_3d_direction)
        #     N_retained_values += N_retained_values_direction
        return tf_3d, N_retained_values

    def compute_wiener_energy(self, i, j, tf_3d_basic):
        """Compute Wiener Energy and store it in self.wiener_energies_ij

        Args:
            :param j:
            :param i:
            :param tf_3d_basic:
        """
        # Formula (8)
        t = np.abs(tf_3d_basic) ** 2
        W_S_wie_xR = t / (t + self.sigma ** 2)
        self.wiener_energies[i, j] = np.mean(W_S_wie_xR, axis=0)  # Store Result

    def wiener_filter(self, tf_3d, i, j):
        # Formula (9)
        # wiener energy is in self.wiener_energies[i, j]
        filtered = np.array([self.wiener_energies[i, j] * slice_2d for slice_2d in tf_3d])
        return filtered

    def weight_th(self, N_retained_values):
        # Formula (10)
        if N_retained_values >= 1:
            w_ht_xR = 1 / self.sigma ** 2 * N_retained_values
        else:
            w_ht_xR = 1
        return w_ht_xR

    def weight_wie(self, i, j):
        """Computes Wiener Coefficient of the basic estimate images for pixel (i,j)

        Args:
            i ([int]): [pixel index]
            j ([int]): [pixel index]

        Returns:
            [float]: [Wiener Coefficient]
        """
        # Formula (11)

        wiener_coeff_ij = (self.sigma * np.linalg.norm(self.wiener_energies[i, j])) ** (-2)

        return wiener_coeff_ij

    def compute_y_basic(self):
        # Formula (12)
        # self.img_basic_estimate = ...
        img = np.zeros((self.N, self.N))
        for i, j in tqdm.tqdm(product(range(self.N), repeat=2)):
            num = 0
            denom = 0
            for ii, jj in product(range(self.N), repeat=2):
                group_3d = self.th_itf_3d[ii, jj]
                values = group_3d.all_values(i, j)
                num += self.w_th[ii, jj] * np.sum(values)
                denom += self.w_th[ii, jj] * len(values)

            if denom != 0:
                img[i, j] = num / denom

        # padding :
        self.img_basic_estimate[:img.shape[0], :img.shape[1]] = img

    def compute_y_final(self):
        # Formula (12)
        # TODO
        # self.img_final_estimate = ...
        for i, j in tqdm.tqdm(product(range(self.N), repeat=2)):
            num = 0
            denom = 0
            for ii, jj in product(range(self.N), repeat=2):
                group_3d = self.wie_itf_3d[ii, jj]
                values = group_3d.all_values(i, j)
                num += self.w_wie[ii, jj] * np.sum(values)
                denom += self.w_wie[ii, jj] * len(values)

            if denom != 0:
                self.img_final_estimate[i, j] = num / denom


params = {
    "N1_th": 4,
    "N1_wie": 4,
    "Ns": 2,
    "N_step": 4,
    "sigma": 2,
    "lambda_3d": 1,
    "lambda_2d": 1,
    "tau_ht_match": 1,
    "tau_wie_match": 1
}
