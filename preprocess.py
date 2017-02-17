#!env/bin/python3
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np


def randomly_orientate(i):
    """Rotates an image by 90 degrees, 50% of the time"""
    return np.rot90(i, 1) if np.random.random() < 0.5 else i


class Preprocessor:
    def __init__(self, preview_dir='data/preview/', train_dir='data/train/',
                 test_dir='data/test/'):
        # init directories
        self.preview_dir = preview_dir
        self.train_dir = train_dir
        self.test_dir = test_dir

        # init data generator for training data
        shift = 0.05
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=5,
            shear_range=0.09,
            width_shift_range=shift,
            height_shift_range=shift,
            preprocessing_function=randomly_orientate,
            fill_mode='nearest'
        )

        # init data generator for testing data
        self.test_datagen = ImageDataGenerator(rescale=1./255)

    def preview_preprocessed_img(self, img_path, n):
        """
        Generates *n* preprocessed copies of *img* and saves them all
        to the *data/preview* directory.  (Useful for debugging.)
        :param img_path: string path to the image.
        :param n: int number of preprocessed copies to output.
        """
        # clear out preview directory
        for f in os.listdir(self.preview_dir):
            os.remove(self.preview_dir + f)

        # load test image into np array
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # generate 1 batch of n images, save to preview directory
        i = 0
        for batch in self.train_datagen.flow(x, batch_size=1,
                                             save_to_dir=self.preview_dir,
                                             save_prefix='preview',
                                             save_format='jpeg'):
            i += 1
            if i >= n:
                break

    def get_train_generator(self):
        """
        Returns the training data generator so that it can be plugged
        into a model directly.
        """
        return self.train_datagen.flow_from_directory(
            self.train_dir,  # S3 data should be mounted here
            batch_size=32,
            class_mode='binary'  # for now, later it'll be categorical
        )

    def get_test_generator(self):
        """
        Returns the testing data generator so that it can be plugged
        into a model directly.
        """
        return self.test_datagen.flow_from_directory(
            self.test_dir,  # S3 data should be mounted here
            batch_size=32,
            class_mode='binary'  # for now, later it'll be categorical
        )

if __name__ == '__main__':
    print('Generating preview images...')
    pp = Preprocessor()
    pp.preview_preprocessed_img('data/train/doc.jpg', 10)
    print('Done.')
