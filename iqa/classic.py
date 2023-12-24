import skimage


def signal_to_noise_ratio(image):
    signal = image.mean()
    noise = skimage.restoration.estimate_sigma(image, average_sigmas=True, channel_axis=2)
    return signal / noise


def ssim(image1, image2):
    skimage.metrics.structural_similarity(
        image1, image2, win_size=11, gaussian_weights=False,
        data_range=max(image1.max() - image1.min(), image2.max() - image2.min()),
        multichannel=False
    )


def multiscale_ssim(image1, image2, num_scales=5):
    mcs_list = ()
    for i in range(num_scales):
        scale = 1 / 2 ** i
        sim = ssim(
            skimage.transform.rescale(image1, scale, anti_aliasing=True),
            skimage.transform.rescale(image2, scale, anti_aliasing=True)
        )
    mcs_list.append(sim)
    return mcs_list