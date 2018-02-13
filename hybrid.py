import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
	k_row=(kernel.shape(0)-1)/2
	k_col=(kernel.shape(1)-1)/2
	
	kernel.shape=(1,len(kernel.flatten()))
	kernel=np.transpose(kernel)
	img_d=img.ndim
	correlation_img=np.zeros((img.shape[0],img.shape[1]),dtype=float)
	
	if img_d==2:
		img=img.reshape((img.shape[0],img.shape[1],1))
    
    for d in range(img.shape[2]):
    	img_gray=img[:,:,d]

    	for i in range(k_row,img_gray.shape[0]-k_row):
        	for j in range(k_col,img_gray.shape[1]-k_col):
	            g=[]
	            img_gray_s=img_gray[i-k_row:i+k_row+1,j-k_col:j+k_col+1]
	            img_gray_s=img_grays.flatten()
	            
	            g=np.matrix.dot(img_gray_s,kernel)
	            correlation_img[i,j,d]=np.sum(g)

	 return correlation_img
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
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
	kernel_trans=kernel.transpose()

	retun cross_correlation_2d(img,kernel_trans)
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
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, width, height):
	w_h=int((width-1)/2)
	h_h=int((height-1)/2)
	pi=np.pi
	
	a=np.array([[x**2+y**2 for x in range(-w_h,w_h+1)] for y in range(-h_h, h_h+1)])
	
	GaussianMatrix=1/(2*pi*sigma**2)*np.exp(-a/(2*sigma**2))

	return GaussianMatrix

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
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
	Gaussian_kernel=gaussian_blur_kernel_2d(sigma,size,size)
	low_pass_img=convolve_2d(img,Gaussian_kernel)
	return low_pass_img

    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
	low_pass_img=low_pass(img, sigma, size)
	high_pass_img=img-low_pass_img
	return high_pass_img
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

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


