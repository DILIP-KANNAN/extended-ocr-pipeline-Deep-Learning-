
import matplotlib.pyplot as plt
import numpy as np

def show_image(img_tensor, title=None):
    # img_tensor expected as HxW or 1xHxW or HxW numpy
    if hasattr(img_tensor, 'detach'):
        img = img_tensor.cpu().numpy()
    else:
        img = np.array(img_tensor)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    plt.figure(figsize=(6,2))
    plt.imshow(img, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
