import onnxruntime as ort
import numpy as np
import cv2
onnx_model_path = './onnxfile'
session = ort.InferenceSession(onnx_model_path)

#input test model
# load spoof img
file_name = './spoof/images'

#load img
img = cv2.imread(file_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (92,92))
img = (img/255.).astype(np.float32)
img = img.transpose((2,0,1))

#transform img
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img[0] = (img[0] - mean[0]) / std[0] 
img[1] = (img[1] - mean[1]) / std[1] 
img[2] = (img[2] - mean[2]) / std[2] 

output_names = [session.get_outputs()[0].name]
input_name = session.get_inputs()[0].name
outputs = session.run(output_names, {input_name: img[np.newaxis,:]})
print("Output:", outputs[0])  # 0보다 작으면 spoofing