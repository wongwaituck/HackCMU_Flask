#!/usr/bin/python
import tensorflow as tf
from os import listdir, remove
from os.path import isfile, join
from sklearn.metrics import accuracy_score
from flask import *
from flask_cors import CORS
import array
import random
import numpy as np
import cv2
from PIL import Image
app = Flask(__name__)

CORS(app)

FILEPATH = 'testData/female/'

sess = tf.Session()
label_lines = ['bad', 'good']
#with tf.gfile.FastGFile('80-6.pb', 'rb') as f:
with tf.gfile.FastGFile('best.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def predict(img_path, name, width, height):
    predictionList = []
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    t_input = sess.graph.get_tensor_by_name('input:0')
    photo = tf.gfile.FastGFile(img_path + name, 'rb').read()
    jpg = tf.image.decode_jpeg(photo, channels=3)
    #jpg = tf.image.resize_image_with_crop_or_pad(jpg, 224, 224)
    #jpg = tf.stack(jpg)
    jpg = tf.reshape(jpg, [1, 224, 224, 3]) 
    #print(jpg.get_shape().as_list())
    #a = np.append([], jpg)
    #jpg = tf.stack(a)
    #print(jpg.get_shape().as_list())

    fin = tf.image.convert_image_dtype(jpg,  tf.float32)
    
    predictions = sess.run(softmax_tensor, {t_input:fin.eval(session=sess)})
   
    print(predictions)    
 
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    prediction = label_lines[top_k[0]]
    predictionList.append(prediction)
    accuracy = (1 - predictions[0][0])
    return accuracy

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/rateme', methods=['POST'])
def rateme():
    rdata = request.get_json()
    # we need to access the pixels, width and height
    pixels = rdata['pixels']
    width = rdata['width']
    height = rdata['height']

    zebytes = array.array('B', pixels).tostring()

    img = Image.frombytes('RGBA', (width, height), zebytes)
    rgb_im = img.convert('RGB') 
    name = str(random.randint(0, 99999)) + '.jpg'  
    rgb_im = np.array(rgb_im) 
    rgb_im = cv2.resize(rgb_im, (224, 224), cv2.INTER_LINEAR)
    rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2BGR)
    #rgb_im.save(FILEPATH + name)
    cv2.imwrite(FILEPATH + name, rgb_im)
    accuracy = predict(FILEPATH, name, width, height)
    
    data = {}
    data['status'] = True
    data['rating'] = accuracy
    return jsonify(data)
