# Paper Making a “Completely Blind” Image Quality Analyzer
# Code https://github.com/guptapraful/niqe/blob/master/niqe.py

import math
import numpy as np
import scipy


gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0/gamma_range)
a *= a
b = scipy.special.gamma(1.0/gamma_range)
c = scipy.special.gamma(3.0/gamma_range)
prec_gammas = a/(b*c)

class NIQE():
    def __init__(self, param_path):
        params = scipy.io.loadmat(param_path)
        self.pop_mu = np.raval(params['pop_mu'])
        self.pop_cov = params['pop_cov']
        self.patch_size = 96
    
    def __call__(self, image):
        features = self.get_patch_features(image)
        sample_mu = np.mean(features, axis=0)
        sample_cov = np.cov(features.T)
        X = sample_mu - self.pop_mu
        covmat = (self.pop_cov + sample_cov) / 2.0
        pinvmat = scipy.linalg.pinv(covmat)
        score = np.sqrt(np.dot(np.dot(X, pinvmat), X))
        return score

    def get_patch_features(self, image):
        h, w = image.shape
        assert(h >= self.patch_size and w >= self.patch_size)

        image = image.astype(np.float32)
        mscn, var, mu = self.compute_image_mscn_transform(image)
        mscn = mscn.astype(np.float32)
        feats = self.extract_on_patches(mscn)

        downscaled_image = scipy.misc.imresize(image, 0.5, interp='bicubic', mode='F')
        downscaled_mscn, _, _ = self.compute_image_mscn_transform(downscaled_image)
        downscaled_mscn = downscaled_mscn.astype(np.float32)
        downscaled_feats = self.extract_on_patches(downscaled_mscn)

        feats = np.hstack((feats, downscaled_feats))
        return feats
    
    def compute_image_mscn_transform(self, image, C=1, avg_window=None, extend_mode='constant'):
        if avg_window is None:
            avg_window = self.gen_gauss_window(3, 7.0/6.0)
        assert len(np.shape(image)) == 2
        h, w = np.shape(image)
        mu_image = np.zeros((h, w), dtype=np.float32)
        var_image = np.zeros((h, w), dtype=np.float32)
        image = np.array(image).astype('float32')
        scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
        scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
        scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
        scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
        var_image = np.sqrt(np.abs(var_image - mu_image**2))
        return (image - mu_image)/(var_image + C), var_image, mu_image

    def gen_gauss_window(self, lw, sigma):
        sd = np.float32(sigma)
        lw = int(lw)
        weights = [0.0] * (2 * lw + 1)
        weights[lw] = 1.0
        sum = 1.0
        sd *= sd
        for ii in range(1, lw + 1):
            tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
            weights[lw + ii] = tmp
            weights[lw - ii] = tmp
            sum += 2.0 * tmp
        for ii in range(2 * lw + 1):
            weights[ii] /= sum
        return weights

    def extract_on_patches(self, img, patch_size):
        h, w = img.shape
        patches = []
        for j in range(0, h - patch_size + 1, patch_size):
            for i in range(0, w - patch_size + 1, patch_size):
                patch = img[j:j+patch_size, i:i+patch_size]
                patches.append(patch)

        patches = np.array(patches)
        
        patch_features = []
        for p in patches:
            patch_features.append(self._niqe_extract_subband_feats(p))
        patch_features = np.array(patch_features)

        return patch_features

    def _niqe_extract_subband_feats(self, mscncoefs):
        # alpha_m,  = extract_ggd_features(mscncoefs)
        alpha_m, N, bl, br, lsq, rsq = self.aggd_features(mscncoefs.copy())
        pps1, pps2, pps3, pps4 = self.paired_product(self, mscncoefs)
        alpha1, N1, bl1, br1, lsq1, rsq1 = self.aggd_features(pps1)
        alpha2, N2, bl2, br2, lsq2, rsq2 = self.aggd_features(pps2)
        alpha3, N3, bl3, br3, lsq3, rsq3 = self.aggd_features(pps3)
        alpha4, N4, bl4, br4, lsq4, rsq4 = self.aggd_features(pps4)
        return np.array([alpha_m, (bl+br)/2.0,
                alpha1, N1, bl1, br1,  # (V)
                alpha2, N2, bl2, br2,  # (H)
                alpha3, N3, bl3, bl3,  # (D1)
                alpha4, N4, bl4, bl4,  # (D2)
        ])

    def paired_product(self, new_im):
        shift1 = np.roll(new_im.copy(), 1, axis=1)
        shift2 = np.roll(new_im.copy(), 1, axis=0)
        shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
        shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

        H_img = shift1 * new_im
        V_img = shift2 * new_im
        D1_img = shift3 * new_im
        D2_img = shift4 * new_im

        return (H_img, V_img, D1_img, D2_img)
    
    def aggd_features(self, imdata):
        #flatten imdata
        imdata.shape = (len(imdata.flat),)
        imdata2 = imdata*imdata
        left_data = imdata2[imdata<0]
        right_data = imdata2[imdata>=0]
        left_mean_sqrt = 0
        right_mean_sqrt = 0
        if len(left_data) > 0:
            left_mean_sqrt = np.sqrt(np.average(left_data))
        if len(right_data) > 0:
            right_mean_sqrt = np.sqrt(np.average(right_data))

        if right_mean_sqrt != 0:
            gamma_hat = left_mean_sqrt / right_mean_sqrt
        else:
            gamma_hat = np.inf
        #solve r-hat norm

        imdata2_mean = np.mean(imdata2)
        if imdata2_mean != 0:
            r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata2))
        else:
            r_hat = np.inf
        rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1)*(gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

        #solve alpha by guessing values that minimize ro
        pos = np.argmin((prec_gammas - rhat_norm)**2);
        alpha = gamma_range[pos]

        gam1 = scipy.special.gamma(1.0/alpha)
        gam2 = scipy.special.gamma(2.0/alpha)
        gam3 = scipy.special.gamma(3.0/alpha)

        aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
        bl = aggdratio * left_mean_sqrt
        br = aggdratio * right_mean_sqrt

        #mean parameter
        N = (br - bl) * (gam2 / gam1)#*aggdratio
        return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)
    
    def ggd_features(imdata):
        nr_gam = 1 / prec_gammas
        sigma_sq = np.var(imdata)
        E = np.mean(np.abs(imdata))
        rho = sigma_sq / E**2
        pos = np.argmin(np.abs(nr_gam - rho));
        return gamma_range[pos], sigma_sq