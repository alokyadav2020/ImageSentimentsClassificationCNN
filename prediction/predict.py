import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import cv2

import os,sys

class ImageSentiments:
    def __init__(self,filename):
        self.filename =filename


    def predictionimage(self):
        # load model
        # model = load_model(os.path.join("model", "model_vgg16.h5"))

        try:
            model = load_model(os.path.join("model", "imageclassifier.h5"))

            imagename = self.filename
            img=cv2.imread(imagename)
            resize=tf.image.resize(img,(256,256))
            result=model.predict(np.expand_dims(resize/255,0))
            # # test_image = image.load_img(imagename, target_size = (224,224))
            # test_image = image.load_img(imagename, target_size = (256,256))
            # test_image = image.img_to_array(test_image)
            # test_image = np.expand_dims(test_image, axis = 0)
            # result=model.predict(np.expand_dims(test_image/255,0))
            # result = np.argmax(model.predict(test_image), axis=1)


            print(result)

            if result > 0.5: 
                print(f'Predicted class is sad')
                prediction = 'Image Sentiment -- Sad'
                return [{ "image" : prediction}]
            else:
               print(f'Predicted class is happy')
               prediction = 'Image Sentiment -- Happy'
               return [{ "image" : prediction}]
        except Exception as e:
            raise(sys,e)

