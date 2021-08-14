import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
import time
from torch.cuda.amp import GradScaler
from EarlyStopping import EarlyStopping
from Common_Function import *
from models.MesoNet import  MesoInception4
#############################EVAL##############################
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

set_seeds()
LIST_SELECT = ['IMAGE', 'VIDEO']
LIST_SELECT = ['IMAGE', 'VIDEO']
for SELECT in LIST_SELECT:
    MODE = SELECT
    calculate = False
    EPOCHS = 50
    BATCH_SIZE = 200
    VALID_RATIO = 0.3
    N_IMAGES = 100
    START_LR = 1e-5
    END_LR = 10
    NUM_ITER = 100
    PATIENCE_EARLYSTOP=10

    test_dir, load_dir = '', ''
    if MODE == 'IMAGE':
        test_dir = "/media/data1/mhkim/FAKEVV_hasam/test/FRAMES/real_A_fake_others"
        load_dir = "./MesoInception4_realA_fakeC.pt"
    elif MODE == 'VIDEO':
        test_dir =  "/media/data1/mhkim/FAKEVV_hasam/test/SPECTOGRAMS/real_A_fake_others"
        load_dir = "./MesoInception4_realA_fakeB.pt"

    pretrained_size = 224
    pretrained_means = [0.4489, 0.3352, 0.3106]#[0.485, 0.456, 0.406]
    pretrained_stds= [0.2380, 0.1965, 0.1962]#[0.229, 0.224, 0.225]

    test_transforms = transforms.Compose([
                               transforms.Resize((pretrained_size,pretrained_size)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean = pretrained_means, 
                                                    std = pretrained_stds)
                           ])
    test_data = datasets.ImageFolder(root = test_dir, 
                                     transform = test_transforms)

    print(f'Number of testing examples: {len(test_data)}')

    test_iterator = data.DataLoader(test_data, 
                                    shuffle = True, 
                                    batch_size = BATCH_SIZE)
    model = MesoInception4()
    model.load_state_dict(torch.load('./MesoInception4_realA_fakeC.pt')['state_dict'])
    OUTPUT_DIM = 2
    print(f'OUTPUT_DIM is {OUTPUT_DIM}')

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    model = model.to(device)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=PATIENCE_EARLYSTOP, verbose=True)

    optimizer = optim.Adam(model.parameters(), lr = START_LR)
    best_valid_loss = float('inf')

    print("eval...")

    start_time = time.monotonic()
    def EVAL_classification(model, test_iterator, device):
        label_encoder = LabelEncoder()
        enc = OneHotEncoder(sparse=False)

        y_true=np.zeros((0,2),dtype=np.int8)
        y_pred=np.zeros((0,2),dtype=np.int8)

        model.eval()
        for i, data in enumerate(test_iterator):
            with torch.no_grad():
                in_1 = data[0].to(device)
                _y_pred = model(in_1).cpu().detach()

                integer_encoded = label_encoder.fit_transform(data[1].detach().cpu())
                integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

                onehot_encoded = enc.fit_transform(integer_encoded)
                onehot_encoded = onehot_encoded.astype(np.int8) 

                _y_true = torch.tensor(onehot_encoded)
                _y_true_argmax = _y_true.argmax(1)
                _y_true = np.array(torch.zeros(_y_true.shape).scatter(1, _y_true_argmax.unsqueeze(1),1),dtype=np.int8)
                y_true = np.concatenate((y_true,_y_true))

                a = _y_pred.argmax(1)
                _y_pred = np.array(torch.zeros(_y_pred.shape).scatter(1, a.unsqueeze(1), 1),dtype=np.int8)
                y_pred = np.concatenate((y_pred,_y_pred))

        result = classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=4, output_dict=False, zero_division='warn')    
        print(result)
        print(f'ACC is {accuracy_score(y_true, y_pred)}')

    EVAL_classification(model,test_iterator,device)
    end_time = time.monotonic()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)









