import os
import onnx
import torch
import timm
def clean_state_dict(state_dict: dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict
root = './mobilenetv4/pytorch-image-models/'
path_list = []
for a,b,c in os.walk(root):
    for _c in c:
        if 'simple.onnx' in _c:
            path_list.append(os.path.join(a,_c))

list_done=[]
import shutil
for li in path_list:
    savefolder = os.path.dirname(li)
    if os.path.exists(savefolder+'/tflite_models'):
        continue
    os.system(f'onnx2tf -oiqt -i {li}')
    #-o 파라미터로 특정경로 지정하면 자꾸 variable~~ 오류나서 Default path에서 복사시키도록 해둠
    _from = './saved_model' 
    shutil.copytree(_from, savefolder+'/tflite_models')
    shutil.rmtree(_from)
    list_done.append(li)
    print(li)  
    # break

import pprint
pprint.pprint(list_done)
