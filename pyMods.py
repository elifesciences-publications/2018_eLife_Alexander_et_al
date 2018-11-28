import numpy as np

class Image_Dim():
    """ Image class represents various features of image data dimensionality """

    def __init__(self):
        """ Create a new set of image dimensions """
        self.x_size = 512
        self.y_size = 512
        self.frames = 1
        self.slices = 1
        self.channels = 1

def getImageShape(image, num_chan, dim):
    """ Return image specifications based on orderImageDim output. """
    image_spec = Image_Dim()
    if num_chan == 3:
        if dim == 'xy':
            (image_spec.y_size, image_spec.x_size) = image.shape;
        elif dim =='xyz':
            (image_spec.y_size, image_spec.x_size, image_spec.slices) = image.shape;
        elif dim =='xyt':
            (image_spec.y_size, image_spec.x_size, image_spec.frames) = image.shape;
        elif dim =='xyc':
            (image_spec.y_size, image_spec.x_size, image_spec.channels) = image.shape;
        elif dim =='xytc':
            (image_spec.y_size, image_spec.x_size, image_spec.frames, image_spec.channels) = image.shape;
        elif dim =='xyzc':
            (image_spec.y_size, image_spec.x_size, image_spec.slices, image_spec.channels) = image.shape;
        elif dim =='xyzt':
            (image_spec.y_size, image_spec.x_size, image_spec.slices, image_spec.frames) = image.shape;
        elif dim =='xyztc':
            (image_spec.y_size, image_spec.x_size, image_spec.slices, image_spec.frames, image_spec.channels) = image.shape;
        else:
            print('Dimensions flag passed in wrong format.  Avaiable options are [xy|xyz|xyt|xyc|xytc|xyzc|xyzt|xyztc]');
    else:
        if dim == 'xy':
            (image_spec.y_size, image_spec.x_size) = image.shape;
        elif dim =='xyz':
            (image_spec.y_size, image_spec.x_size, image_spec.slices) = image.shape;
        elif dim =='xyt':
            (image_spec.y_size, image_spec.x_size, image_spec.frames) = image.shape;
        elif dim =='xyc':
            (image_spec.y_size,image_spec.x_size, image_spec.channels) = image.shape;
        elif dim =='xytc':
            (image_spec.y_size, image_spec.x_size, image_spec.frames, image_spec.channels) = image.shape;
        elif dim =='xyzc':
            (image_spec.y_size, image_spec.x_size, image_spec.slices, image_spec.channels) = image.shape;
        elif dim =='xyzt':
            (image_spec.y_size, image_spec.x_size, image_spec.slices, image_spec.frames) = image.shape;
        elif dim =='xyztc':
            (image_spec.y_size, image_spec.x_size, image_spec.slices, image_spec.frames, image_spec.channels) = image.shape;
        else:
            print('Dimensions flag passed in wrong format.  Avaiable options are [xy|xyz|xyt|xyc|xytc|xyzc|xyzt|xyztc]');

    return image_spec

def orderImageDim(image, num_chan, dim):
    """ Reorder image dimensions based on skimage input to get predictable output. """
    if num_chan == 3:
        if dim == 'xy':
            fiximage = image
        elif dim =='xyz':
            fiximage = np.moveaxis(image, 0, -1)
        elif dim =='xyt':
            fiximage = np.moveaxis(image, 0, -1)
        elif dim =='xyc':
            fiximage = image
        elif dim =='xytc':
            fiximage = np.moveaxis(image, 0, 2)
        elif dim =='xyzc':
            fiximage = np.moveaxis(image, 0, 2)
        elif dim =='xyzt':
            fiximage = np.moveaxis(image, [0, 1], [3, 2])
        elif dim =='xyztc':
            fiximage = np.moveaxis(image, [0, 1], [3, 2])
        else:
            print('Dimensions flag passed in wrong format.  Avaiable options are [xy|xyz|xyt|xyc|xytc|xyzc|xyzt|xyztc]');
    else:
        if dim == 'xy':
            fiximage = image
        elif dim =='xyz':
            fiximage = np.moveaxis(image, 0, -1)
        elif dim =='xyt':
            fiximage = np.moveaxis(image, 0, -1)
        elif dim =='xyc':
            fiximage = np.moveaxis(image, 0, -1)
        elif dim =='xytc':
            fiximage = np.moveaxis(image, [0, 1], [2, 3])
        elif dim =='xyzc':
            fiximage = np.moveaxis(image, [0, 1], [2, 3])
        elif dim =='xyzt':
            fiximage = np.moveaxis(image, [0, 1], [2, 3])
        elif dim =='xyztc':
            fiximage = np.moveaxis(image, [0, 1, 2], [3, 2, 4])
        else:
            print('Dimensions flag passed in wrong format.  Avaiable options are [xy|xyz|xyt|xyc|xytc|xyzc|xyzt|xyztc]');

    return fiximage

