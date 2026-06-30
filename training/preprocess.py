import numpy as np
import matplotlib.image as mpimg


class ImagePreprocessor:

    def __init__(self,
                 image_size=(28,28)):

        self.image_size = image_size

    def rgb_to_gray(self,image):

        if image.ndim == 3:

            return (

                0.299*image[:,:,0]

                +

                0.587*image[:,:,1]

                +

                0.114*image[:,:,2]

            )

        return image

    def resize(self,image):

        h,w=image.shape

        target_h,target_w=self.image_size

        rows=np.linspace(
            0,
            h-1,
            target_h
        ).astype(int)

        cols=np.linspace(
            0,
            w-1,
            target_w
        ).astype(int)

        return image[np.ix_(rows,cols)]

    def normalize(self,image):

        if image.max()>1:

            image=image/255.0

        return image.astype(np.float32)

    def process(self,image_path):

        image=mpimg.imread(image_path)

        image=self.rgb_to_gray(image)

        image=self.resize(image)

        image=self.normalize(image)

        return image