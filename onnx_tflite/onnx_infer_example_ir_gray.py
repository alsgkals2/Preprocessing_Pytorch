import onnxruntime as ort
import numpy as np
import cv2
import os
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
onnx_model_path = './onnx'
session = ort.InferenceSession(onnx_model_path)
    
folder_name = '/foldname'
list_file = []
for a,b,c in os.walk(folder_name):
    for _c in c:
        if ('.jpg' in _c or '.png' in _c) and 'alpha' in _c and '_ir' in _c:
            list_file.append(os.path.join(a,_c))

for file_name in list_file:
    #load img
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #concat one channel image
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    img = cv2.resize(img, (92,92))
    img = (img/255.).astype(np.float32)
    img = img.transpose((2,0,1))

    #transform img as gray
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img[0] = (img[0] - mean[0]) / std[0] 
    img[1] = (img[1] - mean[1]) / std[1] 
    img[2] = (img[2] - mean[2]) / std[2] 

    output_names = [session.get_outputs()[0].name]
    input_name = session.get_inputs()[0].name
    outputs = session.run(output_names, {input_name: img[np.newaxis,:]})
    print(file_name)
    print("Output:", outputs[0], 'FACE' if outputs[0]<0 else 'MASK')  
