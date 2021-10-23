import tensorflow as tf
from PIL import Image
import numpy as np

IMG_HEIGHT = 300
IMG_WIDTH = 300

LABELS = {'abyssinian': 0,
 'balinese': 1,
 'bengal': 2,
 'birman': 3,
 'bombay': 4,
 'burmese': 5,
 'burmilla': 6,
 'calico': 7,
 'chartreux': 8,
 'chausie': 9,
 'cornish rex': 10,
 'devon rex': 11,
 'egyptian mau': 12,
 'havana': 13,
 'himalayan': 14,
 'korat': 15,
 'maine coon': 16,
 'munchkin': 17,
 'norwegian forest cat': 18,
 'ocicat': 19,
 'persian': 20,
 'pixiebob': 21,
 'ragdoll': 22,
 'russian blue': 23,
 'scottish fold': 24,
 'selkirk rex': 25,
 'siamese': 26,
 'siberian': 27,
 'singapura': 28,
 'snowshoe': 29,
 'somali': 30,
 'sphynx': 31,
 'tabby': 32,
 'torbie': 33,
 'tortoiseshell': 34,
 'turkish angora': 35,
 'tuxedo': 36}

BREEDS = list(LABELS.keys())

def image_to_array(input_img, img_width, img_height):
    img = Image.open(input_img)
    img = img.resize((img_width, img_height), resample=0, box=None)
    img = np.array(img)
    img = img / 255.0
    return np.array([img])

class ModelProcessing:
    def __init__(self, model_dir, input_img):
        self.model = tf.keras.models.load_model(model_dir)
        self.input_img = image_to_array(input_img, IMG_WIDTH, IMG_HEIGHT)
        self.img = input_img

    def breed_probabilities(self, top=3):
        self.pred = self.model.predict(self.input_img)
        temp = dict(zip(BREEDS, list(self.pred[0])))
        probabilities = {key: val for key, val in sorted(temp.items(), key = lambda ele: ele[1], reverse = True)}
        prob_dist = {k: round((v*100), 4) for k, v in (list(probabilities.items())[:top])}
        print(prob_dist)
        return prob_dist

    def get_size(self):
        i = Image.open(self.img)
        return i.size
