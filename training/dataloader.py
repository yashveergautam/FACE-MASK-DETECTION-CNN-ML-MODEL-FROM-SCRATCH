import os
import numpy as np

from training.preprocess import ImagePreprocessor


class DataLoader:

    def __init__(self,
                 dataset_path,
                 image_size=(28,28)):

        self.dataset_path=dataset_path

        self.processor=ImagePreprocessor(image_size)

    def load(self):

        X=[]

        y=[]

        folders=sorted(

            [

                f

                for f in os.listdir(self.dataset_path)

                if os.path.isdir(

                    os.path.join(

                        self.dataset_path,

                        f

                    )

                )

            ]

        )

        num_classes=len(folders)

        for class_index,folder in enumerate(folders):

            folder_path=os.path.join(

                self.dataset_path,

                folder

            )

            for image_name in os.listdir(folder_path):

                if image_name.lower().endswith(

                    (".png",".jpg",".jpeg")

                ):

                    image=self.processor.process(

                        os.path.join(

                            folder_path,

                            image_name

                        )

                    )

                    X.append(image)

                    y.append(class_index)

        X=np.array(X)

        X=X.reshape(

            -1,

            1,

            28,

            28

        )

        labels=np.zeros(

            (

                len(y),

                num_classes

            )

        )

        labels[

            np.arange(len(y)),

            y

        ]=1

        return X,labels