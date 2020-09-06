import cv2
import numpy as np
import math as m
from matplotlib import pyplot as plt


def plot_pics( image_list, num_in_col=2, title_list=[]):
    '''
    Create a tabular plot of the images in the list.  The number of
    columns, defaulting to 2, is given by the second argument and the
    optional titles associated with the images are given in the third.
    '''
    if len(image_list) == 0: return
    if len(image_list[0].shape) == 2:
        plt.gray()
    num_rows = int( m.ceil(len(image_list)/float(num_in_col)))
    '''
    if num_in_col > 2:
        plt.figure(figsize(10,10))
    else:
        plt.figure(figsize(15,15))
    '''
    for i in range(len(image_list)):
        im = image_list[i]
        #print num_rows, num_in_col, i+1
        plt.subplot(num_rows, num_in_col, i+1)
        plt.imshow(im)
        if i < len(title_list):
            plt.title(title_list[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def show_with_pixel_values(im):
    '''
    Create a plot of a single image so that pixel intensity or color
    values are shown as the mouse is moved over the image.
    '''
    fig, ax = plt.subplots()
    ax.imshow(im)
    numrows, numcols = im.shape[0], im.shape[1]
    if len(im.shape) == 3:
        plt.gray()

    def format_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if col>=0 and col<numcols and row>=0 and row<numrows:
            z = im[row,col]
            if type(z) is np.ndarray:    # color image
                return '(%d,%d): [%d,%d,%d]' %(col, row, int(z[0]), int(z[1]), int(z[2]))
            else:                        # grayscale image
                return '(%d,%d): %d' %(x, y, int(z))
        else:
            return '(%d,%d)' %(x, y)

    ax.format_coord = format_coord     
    plt.show()


def show_gray_image_with_lines( im, x1, y1,x2, y2 ):
    '''
    Show an image with line segments drawn on top of it.  Each of
    x1,y1, x2,y2 are assumed (without checking) to be the same length
    and each entry determines the endpoints of a line segment to be
    drawn on the image.
    '''
    plt.gray()
    plt.imshow(im)
    for i in range(len(x1)):
        x_list = [ x1[i], x2[i] ]
        y_list = [ y1[i], y2[i] ]
        plt.plot(x_list, y_list)
    plt.show()


if __name__ == "__main__":
    import sys
    img_list = []
    name_list = []
    for im_name in sys.argv[1:]:
        im = cv2.imread(im_name)
        if len(im.shape) == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            img_list.append( im )
            name_list.append( im_name )
    #print name_list
    num_in_col = 2
    plot_pics( img_list, num_in_col, name_list )

    
    show_with_pixel_values( img_list[0] )
    gray_im = cv2.cvtColor(img_list[0], cv2.COLOR_RGB2GRAY )
    show_with_pixel_values( gray_im )
            

