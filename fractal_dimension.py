import cv2
import numpy as np
from skimage.color import rgb2gray

def compute_fractal_dimension(image, threshold=0.9):
    """
    To find the fractal dimension of an image using box-counting.
    
    Parameters:
      image (np.ndarray): Input image (RGB or grayscale).
      threshold (float): Threshold to binarize the image.
    
    Returns:
      float: The fractal dimension.
    """
    
    if len(image.shape) > 2:
        image = rgb2gray(image)
    
    img = image < threshold

  
    p = min(img.shape)
   
    img = img[:p, :p]

    
    sizes = np.logspace(np.log10(2), np.log10(p/2), num=10, dtype=int)
    sizes = np.unique(sizes)
    
    counts = []
    for size in sizes:
        
        count = 0
        for i in range(0, p, size):
            for j in range(0, p, size):
                if np.any(img[i:i+size, j:j+size]):
                    count += 1
        counts.append(count)

    
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

if __name__ == '__main__':
   
    import sys
    from PIL import Image
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = np.array(Image.open(img_path))
        fd = compute_fractal_dimension(img)
        print("Fractal Dimension:", fd)
    else:
        print("Please provide an image path.")
