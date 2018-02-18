import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    kernel_width = kernel.shape[0]
    kernel_height = kernel.shape[1]

    padImg = np.pad(img, ((kernel_width - 1) / 2, (kernel_height - 1 ) / 2), 'constant')

    if img.ndim == 2:
        newImg = np.zeros(img.shape)
        for i in range(padImg.shape[0] - kernel_width + 1):
            for j in range(padImg.shape[1] - kernel_height + 1):
                newImg[i,j] = np.sum(padImg[i:i+kernel_width, j:j+kernel_height]*kernel)
        return newImg

    if img.ndim == 3:
        newImg = np.zeros(img.shape)
        for k in range(3):
            for i in range(padImg.shape[0] - kernel_width + 1):
                for j in range(padImg.shape[1] - kernel_height + 1):
                    newImg[i,j,k] = np.sum(padImg[i:i+kernel_width, j:j+kernel_height, k]*kernel)
        return newImg


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return cross_correlation_2d(img, kernel[::-1, ::-1])

def gaussian_blur_kernel_2d(sigma, width, height):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    '''

    width_mean = (width - 1) / 2
    height_mean = (height -1) / 2

    double kernel[width][height];
    for (int x = 0; x < width; ++x)
        for (int y = 0; y < height; ++y) {
            kernel[x][y] = np.exp( -(np.power((width-width_mean), 2.0) + np.power((height-height_mean),2.0)) / float ( 2 * (sigma ** 2)) )
        }

    kernel = kernel / kernel.sum()

    return kernel

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    lowPassImg = convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))
    return lowPassImg

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    newImg = img - low_pass(img, sigma, size)
    return newImg

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
