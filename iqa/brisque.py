# Paper: No-Reference Image Quality Assessment in the Spatial Domain
# code: https://github.com/rehanguha/brisque/blob/master/brisque/brisque.py

from libsvm import svmutil
import numpy as np
import pickle
import scipy
import skimage


class BRISQUE():
    def __init__(self, model_path, norm_path, kernel_size=7, sigma=7 / 6):
        self.model = svmutil.svm_load_model(model_path)
        with open(norm_path, 'rb') as f:
            self.scale_params = pickle.load(f)
        self.kernel_size = kernel_size
        self.sigma = sigma
        # gaussian kernel
        Y, X = np.indices((kernel_size, kernel_size)) - int(kernel_size / 2)
        gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
        self.gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    
    def __call__(self, image):
        image = np.array(image)
        gray_image = skimage.color.rgb2gray(image)
        features = self.get_features(gray_image)
        downscaled_image = skimage.transform.rescale(gray_image, 0.5, order=3, anti_aliasing=True)
        downscaled_features = self.get_features(downscaled_image)
        features = np.concatenate((features, downscaled_features))
        score = self.get_image_quality_score(features)
        return score
        
    def get_features(self, image):
        local_mean = scipy.signal.convolve2d(image, self.gaussian_kernel, 'same')
        E_square = scipy.signal.convolve2d(image ** 2, self.gaussian_kernel, 'same')
        local_var = np.sqrt(np.abs(local_mean ** 2 - E_square))
        mscn_coefficients = (image - local_mean) / (local_var + 1 / 255)
        horizontal_coefficients = mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:]
        vertical_coefficients = mscn_coefficients[:-1, :] * mscn_coefficients[1:, :]
        first_diagonal_coefficients = mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:]
        second_diagonal_coefficients = mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
        mscn_features = self.generalized_gaussian_fit(mscn_coefficients)
        horizontal_features = self.asymmetric_generalized_gaussian_fit(horizontal_coefficients)
        vertical_features = self.asymmetric_generalized_gaussian_fit(vertical_coefficients)
        first_diagonal_features = self.asymmetric_generalized_gaussian_fit(first_diagonal_coefficients)
        second_diagonal_features = self.asymmetric_generalized_gaussian_fit(second_diagonal_coefficients)
        features = np.concatenate([mscn_features, horizontal_features, vertical_features, first_diagonal_features, second_diagonal_features])
        return features
    
    def asymmetric_generalized_gaussian_fit(self, x):
        def estimate_phi(alpha):
            numerator = scipy.special.gamma(2 / alpha) ** 2
            denominator = scipy.special.gamma(1 / alpha) * scipy.special.gamma(3 / alpha)
            return numerator / denominator

        def estimate_r_hat(x):
            size = np.prod(x.shape)
            return (np.sum(np.abs(x)) / size) ** 2 / (np.sum(x ** 2) / size)

        def estimate_R_hat(r_hat, gamma):
            numerator = (gamma ** 3 + 1) * (gamma + 1)
            denominator = (gamma ** 2 + 1) ** 2
            return r_hat * numerator / denominator

        def mean_squares_sum(x, filter=lambda z: z == z):
            filtered_values = x[filter(x)]
            squares_sum = np.sum(filtered_values ** 2)
            return squares_sum / ((filtered_values.shape))

        def estimate_gamma(x):
            left_squares = mean_squares_sum(x, lambda z: z < 0)
            right_squares = mean_squares_sum(x, lambda z: z >= 0)

            return np.sqrt(left_squares) / np.sqrt(right_squares)

        def estimate_alpha(x):
            r_hat = estimate_r_hat(x)
            gamma = estimate_gamma(x)
            R_hat = estimate_R_hat(r_hat, gamma)

            solution = scipy.optimize.root(lambda z: estimate_phi(z) - R_hat, [0.2]).x

            return solution[0]

        def estimate_sigma(x, alpha, filter=lambda z: z < 0):
            return np.sqrt(mean_squares_sum(x, filter))

        def estimate_mean(alpha, sigma_l, sigma_r):
            return (sigma_r - sigma_l) * constant * (scipy.special.gamma(2 / alpha) / scipy.special.gamma(1 / alpha))

        alpha = estimate_alpha(x)
        sigma_l = estimate_sigma(x, alpha, lambda z: z < 0)
        sigma_r = estimate_sigma(x, alpha, lambda z: z >= 0)

        constant = np.sqrt(scipy.special.gamma(1 / alpha) / scipy.special.gamma(3 / alpha))
        mean = estimate_mean(alpha, sigma_l, sigma_r)

        return alpha, mean, sigma_l, sigma_r
    
    def generalized_gaussian_fit(self, x):
        alpha, mean, sigma_l, sigma_r = self.asymmetric_generalized_gaussian_fit(x)
        var = (sigma_l ** 2 + sigma_r ** 2) / 2
        return alpha, var
    
    def get_image_quality_score(self, features):
        min_ = np.array(self.scale_params['min_'])
        max_ = np.array(self.scale_params['max_'])
        scaled_features = (features - min_) / (max_ - min_) * 2 - 1

        x, idx = svmutil.gen_svm_nodearray(
            scaled_features, isKernel=(self.model.param.kernel_type == svmutil.PRECOMPUTED)
        )
        prob_estimates = svmutil.c_double * 1
        return svmutil.libsvm.svm_predict_probability(self.model, x, prob_estimates)