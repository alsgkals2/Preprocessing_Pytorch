
# imageio (mor comfortable)
import imageio #pip install imageio[ppmpeg]
import cv2
from matplotlib import pyplot as plt

for _mp in mv_files[1:]:
    save_path = os.path.dirname(_mp).replace('AntiSpoofing', 'mhkim')
    os.makedirs(save_path, exist_ok=True)
    file = os.path.basename(_mp).split('.')[0]#replace('.mov', '.jpg')
    print(save_path, file)
    reader = imageio.get_reader(_mp, 'ffmpeg')
    print(reader.count_frames())
    for i, im in enumerate(reader):
        print('Mean of frame %i is %1.1f' % (i, im.mean()))
        im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(save_path,file+f'_{i}.jpg'), im)
        plt.imshow(im)
        plt.show()
        break
    break

# opencv version
import os
import cv2
import glob
path = 'path name'
save_path = 'save_path name'
list_mp = glob.glob(os.path.join(path,'*'))

swit = True
for _mp in list_mp:
    print(_mp)

    count = 0
    title = _mp.split('\\')[-1]
    _save_path = os.path.join(save_path,title)
    try:
        os.makedirs(_save_path,exist_ok=False)
    except:
        continue
    
    vidcap = cv2.VideoCapture(_mp)
    ret = True
    while(ret):
    #while(vidcap.isOpened()):
        ret, image = vidcap.read() # read per frame
        # image = cv2.resize(image, (960, 540)) # resizing frame extracted
        if not ret: break
        if(int(vidcap.get(1)) % 10 == 0):
            cv2.imwrite(_save_path+f"/{title}_{count}.png", image) #print('Saved frame%d.jpg' % count)
            count += 1

print("END")

