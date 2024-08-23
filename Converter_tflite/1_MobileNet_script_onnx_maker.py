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

path_list = []
folder = './mobilenetv4/pytorch-image-models/outputs'
for a,b,c in os.walk(folder):
    for _c in c:
        path_list.append(os.path.join(a,_c))

list_done=[]
for li in path_list:
    model_size = 'planemed' if 'planemed' in li else 'med' if 'med' in li else 'large' if 'large' in li else 'small' if 'small' in li else ''
    input_size = '128' if '128' in li else '92' if '92' in li else 0

    if not input_size:
        continue
    num_class = 1
    checkpoint = torch.load(li)
    f = li.replace('.pth.tar', '.onnx').replace('.pth', '.onnx')  # filename
    if os.path.exists(f):
        continue
    #load models
    if 'large' in model_size:
        model = timm.create_model('mobilenetv4_conv_large.e500_r256_in1k', num_classes=num_class, pretrained=True).cpu()
    elif 'planemed' in model_size:
        model = timm.create_model('hf_hub:timm/mobilenetv4_conv_medium.e500_r224_in1k', pretrained=True, num_classes=num_class).cpu() # plane_medium
    elif 'med' in model_size:
        model = timm.create_model('hf_hub:timm/mobilenetv4_conv_blur_medium.e500_r224_in1k', num_classes=num_class, pretrained=True).cpu()
    elif 'small' in model_size:
        model = timm.create_model("hf_hub:timm/mobilenetv4_conv_small.e2400_r224_in1k", num_classes=num_class).cpu()
    try:
        model.load_state_dict(clean_state_dict(checkpoint['state_dict']))
    except Exception as e:
        print(e)
        print('path:', li)
        continue
    model.eval()

    img = torch.zeros((1, 3, int(input_size), int(input_size)), device=next(model.parameters()).device).cpu()  # input
    torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                        output_names=['output'],
                        dynamic_axes=None)

    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    onnx.save(onnx_model, f)
    print('ONNX export success, saved as %s' % f)
    
    #save simple version
    simplify = True
    if simplify:
        try:
            import onnxsim
            print('\nStarting to simplify ONNX...')
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')
        f = f.replace('.onnx', '_simple.onnx')
        onnx.save(onnx_model, f)
        print('ONNX export success, saved as %s' % f)  
    list_done.append(f)
    print(f)

import pprint
pprint.pprint(list_done)
