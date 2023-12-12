import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



def show_img(img, labels, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8 ,8))
    #     img = clahe.apply(img)
    #     plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='bone')

    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0 ,0) ,1 ,1, color=_c) for _c in [(0.667 ,0.0 ,0.0), (0.0 ,0.667 ,0.0), (0.0 ,0.0 ,0.667)]]
        plt.legend(handles, labels)
    plt.axis('off')


def plot_batch(imgs, msks, labels):
    size = len(labels)
    plt.figure(figsize=(5*5, 5))
    for idx in range(size):
        plt.subplot(1, 5, idx+1)
        img = imgs[idx,].permute((1, 2, 0)).numpy()*255.0
        img = img.astype('uint8')
        msk = msks[idx,].permute((1, 2, 0)).numpy()*255.0
        show_img(img, labels, msk)
    plt.tight_layout()
    plt.show()
