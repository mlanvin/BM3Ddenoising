import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

def import_images(size=64):
    """
    Import images from folder ./images
    """
    filelist = glob.glob('images/*.png')
    N = len(filelist)
    imgs = np.zeros((N, size,size))
    for i in range(N):
        raw = cv2.imread(filelist[i], cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(raw, (size,size))
        imgs[i] = resized
    return(imgs, N)

def plot_imbw(img, ax=None, title=""):
    """
    Plot black and white image
    """
    if ax==None:
        plt.figure()
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(title)
        plt.show()
    else:
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        
def split_img(img, block_size=32):
    """
    Return a list of blocks of size block_size x block_size
    """
    m,n = img.shape
    nb_bloc32_x = m//block_size
    nb_bloc32_y = n//block_size
    nb_bloc32 = nb_bloc32_x * nb_bloc32_y
    list_bloc_out = np.zeros((nb_bloc32, block_size, block_size))
    no_bloc = 0
    for i in range(nb_bloc32_x):
        for j in range(nb_bloc32_y):
            list_bloc_out[no_bloc] = img[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]
            no_bloc += 1 
    return list_bloc_out

def snr(img, sigma):
    """
    Computes SNR from image and sigma the std from noise
    """
    m, n = img.shape
    return 20*np.log10(np.linalg.norm(img)/(np.sqrt(m*n)*sigma))

def get_sigma_from_SNR(SNR, img):
    """
    returns the STD needed to get the provided SNR
    """
    m, n = img.shape 
    sigma = (10**(SNR/20)*np.sqrt(m*n)/np.linalg.norm(img))**(-1)
    return(sigma)


def ajout_bruit(img, sigma=1):
    """
    Retourne l'image bruitée en ajoutant un bruit gaussien additif de moyenne nulle et d'écart type sigma fourni
    """
    m, n = img.shape
    bruit = sigma * np.random.randn(m,n)
    img_noisy = img + bruit
    return(img_noisy)
    