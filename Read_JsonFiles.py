"""
json파일은 모두 dictionary 형태
josn.load : 불러오기
json.dump(변수,path) : 저장하기
file['name_dict'] 해서 정보 get 가능
"""

import glob #josn이 여러 개 일 땐 glob함수 사용
import shutil #파일 다룰 땐 본 라이브러리 사용
import json

file_name = '/home/mhkim/AGC/label.json'
with open(file_name) as json_file:
    json_data = json.load(json_file)

ano = list(json_data['annotations'])
listname=[]
#
for a in ano:
    listname.append(a['file_name'].split(' ')[0])
listname = list(set(listname))#중복제거
prevname = listname[0]
print(len(listname))
cnt=0
for name in listname:
    for a in ano:
        if a['box'] and name in a['file_name']:
            a['box']=[]
            del a['box'][:]
    #수정된 josn파일(josn_data) 저장(josn_file)하기
    with open('/home/mhkim/AGC/json_mh/' + str(cnt) + '.json', 'w') as json_file:
        json.dump(json_data, '/home/mhkim/examples/json_mh/' + str(cnt) + '_.json', 'w')