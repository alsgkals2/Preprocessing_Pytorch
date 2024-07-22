import os
print("tdst")
# 주어진 디렉토리에 있는 항목들의 이름을 담고 있는 리스트를 반환합니다.
# 리스트는 임의의 순서대로 나열됩니다.
file_path = '/home/mhkim/AGC_final_add_background/AGC_3class/AGC_2/ClassificationDataset/1'
file_names = os.listdir(file_path)
i = 1
for name in file_names:
    #print(name.split('.')[0]+'_p.jpg')
    #break
    src = os.path.join(file_path, name)
    dst = name.split('.')[0] + '_p.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1