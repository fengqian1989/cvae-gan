import numpy as np
import keras

from .datasets import ConditionalDataset

def load_data():
    labelDictionary = {'kawaii':0,'natural':1,'otona':2, 'brown':3}
    x_train,y_train = load_img_to_numpy('./tags_4_hair','.jpg',labelDictionary)

    datasets = ConditionalDataset()
    datasets.images = x_train
    datasets.attrs = y_train
    datasets.attr_names = ['kawaii','natural','otona', 'brown']

    return datasets

def load_img_to_numpy(dirname,suffix,labelDictionary):
    from pathlib import Path
    from PIL import Image
    numpy_images = []
    label = []
    p = Path(dirname)
    file_dirnames = list(p.glob('**/*'))
    # print(file_dirnames)
    for filename in file_dirnames:
        if filename.suffix == suffix:
            class_name = filename.parts[-2]
            label.append(labelDictionary[class_name])
            raw_img = Image.open(filename).resize((64,64))
            array_img = np.asarray(raw_img)
            array_img.flags.writeable = True
            numpy_images.append(array_img)
    numpy_images = np.asarray(numpy_images,dtype="float32")
    label = np.asarray(label,dtype="float32")
    print(label[0])
    label = keras.utils.to_categorical(label)
    numpy_images /= 255
    return numpy_images,label
