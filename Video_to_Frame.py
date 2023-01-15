
# imageio (mor comfortable)
import imageio
import cv2
from matplotlib import pyplot as plt
import pandas as pd

import glob
import os
#imageio version
def read_face(filename):
  """Reads a single file containing the KeyLemon face locations.

  Parameters:
  filename -- the name of the text file containing the face locations

  Returns:
  A numpy ndarray which is a 2-D integer (integers of 16 bits) array containing
  in every line the frame and every column, the 5 values defined by the
  "README" file in this directory.
  """

  f = open(filename, 'rt') #opens the file for reading

  # we read all lines that are not empty
  lines = [k.strip() for k in f.readlines() if k.strip()]

  # create a numpy matrix filled with zeros and with the right dimensions to
  # hold our data. The element type is set to integers of 16 bits.
  retval = np.zeros((len(lines), 5), dtype='int16')

  # iteratively transform the data in every line and store it on the
  # to-be-returned matrix
  for i, line in enumerate(lines):
    s = line.split()
    for j in range(5):
      retval[i,j] = int(s[j])

  return retval


mv_files = []
root = '/mnt/Face_Private-NFS/AntiSpoofing/Idiap/'
for a,b,c in os.walk(root+'devel'):
    for _c in c :
        if '.mov' in _c:
            mv_files.append(os.path.join(a, _c))
print(len(mv_files))


for _mp in mv_files[1:]:
    save_path = os.path.dirname(_mp).replace('AntiSpoofing', 'abcd')
    os.makedirs(save_path, exist_ok=True)
    file = os.path.basename(_mp).split('.')[0]#replace('.mov', '.jpg')
    print(save_path, file)
    reader = imageio.get_reader(_mp, 'ffmpeg')
    print(reader.count_frames())

    info_frame = pd.DataFrame(read_face(_mp.replace('.mov', '.face').replace('Idiap','Idiap/face-locations')), index=None)#.drop(0,axis=1)
    max_len = len(info_frame)

    for i, im in enumerate(reader):
        if max_len <= i :
            break
        x, y, w, h = list(info_frame.iloc[i])[1:]
        print('Mean of frame %i is %1.1f' % (i, im.mean()))
        im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        im = im[y:y+h, x:x+w, :]
        plt.imshow(im)
        plt.show()
        cv2.imwrite(os.path.join(save_path,file+f'_{i}.jpg'), im)

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

