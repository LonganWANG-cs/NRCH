import numpy as np
import h5py
import pdb 
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import scipy.io as sio
import os
import matplotlib.pyplot as plt

class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


def get_clean_and_noisy_index(dataset,noise_rate):
    if dataset == 'nuswide21':
        noise = h5py.File('./noise/nus-wide-tc21-lall-noise_{}.h5'.format(noise_rate))
    elif dataset == 'flickr':
        noise = h5py.File('./noise/mirflickr25k-lall-noise_{}.h5'.format(noise_rate))
    elif dataset == 'ms-coco':
        noise = h5py.File('./noiseMSCOCO-lall-noise_{}.h5'.format(noise_rate))
    elif dataset == 'iapr':
        noise = h5py.File('./noiseIAPR-lall-noise_{}.h5'.format(noise_rate))
    fl = list(noise['True'])
    ffl = list(noise['result'])
    clean_index = []
    noisy_index = []

    for i in range(len(fl)):
        equal = True
        #pdb.set_trace()
        for j in range(len(fl[i])):
            if fl[i][j] != ffl[i][j]:
                equal=False
        if equal:
            clean_index.append(i)
            
        else:
            noisy_index.append(i)

    #pdb.set_trace()
    return clean_index, noisy_index

class DataList(object):
    def __init__(self, dataset, data_type, transform, noise_type, noise_rate, random_state):
        self.data_type = data_type
        if dataset == 'nuswide21':
            data = h5py.File('./data/NUS-WIDE.h5', 'r')
            noise = h5py.File('./noise/nus-wide-tc21-lall-noise_{}.h5'.format(noise_rate))
        elif dataset == 'flickr':
            data = h5py.File('./data/MIRFlickr.h5', 'r')
            noise = h5py.File('./noise/mirflickr25k-lall-noise_{}.h5'.format(noise_rate))
        elif dataset == 'ms-coco':
            data = h5py.File('./data/MS-COCO.h5', 'r')
            noise = h5py.File('./noise/MSCOCO-lall-noise_{}.h5'.format(noise_rate))
        elif dataset == 'iapr':
            data = h5py.File('./data/IAPR.h5', 'r')
            noise = h5py.File('./noise/IAPR-lall-noise_{}.h5'.format(noise_rate))
        if data_type == "train":
            fi = list(data['ImgTrain'])
            fl = list(data['LabTrain'])
            ffl = list(noise['result'])
            ft = list(data['TagTrain'])
            self.imgs = fi
            self.labs = fl
            self.flabs = ffl
            self.tags = ft
            lab = self.labs[1]
            lab = lab.astype(int)
        elif data_type == "test":
            fi = list(data['ImgQuery'])
            fl = list(data['LabQuery'])
            ft = list(data['TagQuery'])
            self.imgs = fi
            self.labs = fl
            self.tags = ft
        elif data_type == "database":
            fi = list(data['ImgDataBase'])
            fl = list(data['LabDataBase'])
            ft = list(data['TagDataBase'])
            self.imgs = fi
            self.labs = fl
            self.tags = ft
        self.transform = transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.random_state = random_state

    def __getitem__(self, index):
        img = self.imgs[index]
        img = img.astype(np.float32)
        lab = self.labs[index]
        lab = lab.astype(int)
        tlab = lab
        if self.data_type == "train":
            lab = self.flabs[index]
            lab = lab.astype(int)
        tag = self.tags[index]
        tag = tag.astype(np.float32)
        return img, tag, tlab, lab, index

    def __len__(self):
        return len(self.imgs)

def SaveH5File_F(resize_size):
    train_size = 10000
    query_size = 2000
    root = './data/'
    fi = sio.loadmat(os.path.join(root, 'mirflickr25k-iall-vgg.mat'))['XAll']
    fl = sio.loadmat(os.path.join(root, 'mirflickr25k-lall.mat'))['LAll']
    ft = sio.loadmat(os.path.join(root, 'mirflickr25k-yall.mat'))['YAll']
    imgs = list(fi[query_size: query_size + train_size])
    labs = list(fl[query_size: query_size + train_size])
    tags = list(ft[query_size: query_size + train_size])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,24])
    Tag = np.zeros([n,1386])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf = h5py.File('./data/MIRFlickr.h5','w')
    hf.create_dataset('ImgTrain', data = Img)
    hf.create_dataset('TagTrain', data = Tag)
    hf.create_dataset('LabTrain', data = Lab)

    imgs = list(fi[0: query_size])
    labs = list(fl[0: query_size])
    tags = list(ft[0: query_size])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,24])
    Tag = np.zeros([n,1386])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgQuery', data = Img)
    hf.create_dataset('TagQuery', data = Tag)
    hf.create_dataset('LabQuery', data = Lab)

    imgs = list(fi[query_size::])
    labs = list(fl[query_size::])
    tags = list(ft[query_size::])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,24])
    Tag = np.zeros([n,1386])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        #pdb.set_trace()
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgDataBase', data = Img)
    hf.create_dataset('TagDataBase', data = Tag)
    hf.create_dataset('LabDataBase', data = Lab)
    hf.close()

def SaveH5File_C(resize_size):
    train_size = 10000
    query_size = 5000
    root = '../data/'
    path = root + 'MSCOCO_deep_doc2vec_data.h5py'
    data = h5py.File(path)
    fi = np.concatenate([data['train_imgs_deep'][()], data['test_imgs_deep'][()]], axis=0)
    fl = np.concatenate([data['train_imgs_labels'][()], data['test_imgs_labels'][()]], axis=0)
    ft = np.concatenate([data['train_text'][()], data['test_text'][()]], axis=0)
    imgs = list(fi[query_size: query_size + train_size])
    labs = list(fl[query_size: query_size + train_size])
    tags = list(ft[query_size: query_size + train_size])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,80])
    Tag = np.zeros([n,300])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf = h5py.File('../data/MS-COCO.h5','w')
    hf.create_dataset('ImgTrain', data = Img)
    hf.create_dataset('TagTrain', data = Tag)
    hf.create_dataset('LabTrain', data = Lab)

    imgs = list(fi[0: query_size])
    labs = list(fl[0: query_size])
    tags = list(ft[0: query_size])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,80])
    Tag = np.zeros([n,300])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        #pdb.set_trace()
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgQuery', data = Img)
    hf.create_dataset('TagQuery', data = Tag)
    hf.create_dataset('LabQuery', data = Lab)

    imgs = list(fi[query_size::])
    labs = list(fl[query_size::])
    tags = list(ft[query_size::])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,80])
    Tag = np.zeros([n,300])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        #pdb.set_trace()
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgDataBase', data = Img)
    hf.create_dataset('TagDataBase', data = Tag)
    hf.create_dataset('LabDataBase', data = Lab)
    hf.close()            
        
def SaveH5File_N(resize_size):
    train_size = 10500
    query_size = 2100
    root = '../data/'
    fi = sio.loadmat(root + 'nus-wide-tc21-xall-vgg-clean.mat')['XAll']
    fl = sio.loadmat(root + 'nus-wide-tc21-lall-clean.mat')['LAll']
    ft = sio.loadmat(root + 'nus-wide-tc21-yall-clean.mat')['YAll']
    imgs = list(fi[query_size: query_size + train_size])
    labs = list(fl[query_size: query_size + train_size])
    tags = list(ft[query_size: query_size + train_size])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,21])
    Tag = np.zeros([n,1000])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf = h5py.File('../data/NUS-WIDE.h5','w')
    hf.create_dataset('ImgTrain', data = Img)
    hf.create_dataset('TagTrain', data = Tag)
    hf.create_dataset('LabTrain', data = Lab)

    imgs = list(fi[0: query_size])
    labs = list(fl[0: query_size])
    tags = list(ft[0: query_size])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,21])
    Tag = np.zeros([n,1000])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgQuery', data = Img)
    hf.create_dataset('TagQuery', data = Tag)
    hf.create_dataset('LabQuery', data = Lab)

    imgs = list(fi[query_size::])
    labs = list(fl[query_size::])
    tags = list(ft[query_size::])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,21])
    Tag = np.zeros([n,1000])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        #pdb.set_trace()
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgDataBase', data = Img)
    hf.create_dataset('TagDataBase', data = Tag)
    hf.create_dataset('LabDataBase', data = Lab)
    hf.close()     


def SaveH5File_I(resize_size):
    root = '../data/'
    file_path = os.path.join(root, 'iapr-tc12-rand.mat')
    data = sio.loadmat(file_path)
    valid_img = data['VDatabase'].astype('float32')
    valid_txt = data['YDatabase'].astype('float32')
    valid_labels = data['databaseL']
    test_img = data['VTest'].astype('float32')
    test_txt = data['YTest'].astype('float32')
    test_labels = data['testL']
    fi, ft, fl = np.concatenate([valid_img, test_img]), np.concatenate([valid_txt, test_txt]), np.concatenate([valid_labels, test_labels])
    query_size = 2000
    train_size = 10000
    imgs = list(fi[query_size: query_size + train_size])
    labs = list(fl[query_size: query_size + train_size])
    tags = list(ft[query_size: query_size + train_size])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,255])
    Tag = np.zeros([n,2912])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf = h5py.File('../data/IAPR.h5','w')
    hf.create_dataset('ImgTrain', data = Img)
    hf.create_dataset('TagTrain', data = Tag)
    hf.create_dataset('LabTrain', data = Lab)

    imgs = list(fi[0: query_size])
    labs = list(fl[0: query_size])
    tags = list(ft[0: query_size])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,255])
    Tag = np.zeros([n,2912])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgQuery', data = Img)
    hf.create_dataset('TagQuery', data = Tag)
    hf.create_dataset('LabQuery', data = Lab)

    imgs = list(fi[query_size::])
    labs = list(fl[query_size::])
    tags = list(ft[query_size::])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,255])
    Tag = np.zeros([n,2912])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgDataBase', data = Img)
    hf.create_dataset('TagDataBase', data = Tag)
    hf.create_dataset('LabDataBase', data = Lab)
    hf.close()     

def get_data(config):
    dsets = {}
    dset_loaders = {}

    for data_type in ["train", "test", "database"]:
        dsets[data_type] = DataList(config["dataset"], data_type,
                                    transforms.ToTensor(), config["noise_type"], config["noise_rate"], config["random_state"])
        print(data_type, len(dsets[data_type]))
        dset_loaders[data_type] = util_data.DataLoader(dsets[data_type],
                                                      batch_size=config["batch_size"],
                                                      shuffle=True, num_workers=2)

    return dset_loaders["train"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train"]), len(dsets["test"]), len(dsets["database"])


def compute_img_result(dataloader, net, device):
    bs, tclses, clses = [], [], []
    net.eval()
    for img, tag, tcls, cls, _ in tqdm(dataloader):
        tclses.append(tcls)
        clses.append(cls)
        bs.append((net(img.to('cuda'))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def compute_tag_result(dataloader, net, device):
    bs,tclses, clses = [], [], []
    net.eval()
    for img, tag, tcls, cls, _ in tqdm(dataloader):
        tclses.append(tcls)
        clses.append(cls)
        tag = tag.float()
        bs.append((net(tag.to('cuda'))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def CalcHammingDist(B1, B2):
    #B1=B1.cpu()
    #B2=B2.cpu()
    q = B2.shape[1]
    distH = 0.5 * (q - torch.matmul(B1, B2.transpose(0,1)))

    return distH

def calc_map_k( rB, qB, retrieval_label, query_label, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # sim: {0, 1}^{mxn}
    # check if qB is a numpy array and convert it to a torch tensor if it is
    
    if isinstance(qB, np.ndarray):
        qB = torch.from_numpy(qB)
    # check if rB is a numpy array and convert it to a torch tensor if it is
    if isinstance(rB, np.ndarray):
        rB = torch.from_numpy(rB)
    # check if query_label is a numpy array and convert it to a torch tensor if it is
    if isinstance(query_label, np.ndarray):
        query_label = torch.from_numpy(query_label)
    # check if retrieval_label is a numpy array and convert it to a torch tensor if it is
    if isinstance(retrieval_label, np.ndarray):
        retrieval_label = torch.from_numpy(retrieval_label)
    qB, rB, query_label, retrieval_label = [i.cuda().to(dtype=torch.float64) for i in [qB, rB, query_label, retrieval_label]]
    num_query = query_label.shape[0]
    map = 0.
    GND = (query_label.mm(retrieval_label.t()) > 0).type(torch.float).squeeze().cuda()
    if k is None:
        k = retrieval_label.shape[0]
    sum_query =num_query
    for iter in tqdm(range(num_query)):
        # gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        gnd = GND[iter, :]
        tsum = torch.sum(gnd)
        if tsum == 0:
            sum_query-=1
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        # count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        count = torch.arange(1, total + 1).type(torch.float)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count.cuda() / tindex.cuda())
    map = map / sum_query
    return map


def pr_curve(rB, qB, retrieval_L, query_L):
    if isinstance(qB, np.ndarray):
        qB = torch.from_numpy(qB)
    # check if rB is a numpy array and convert it to a torch tensor if it is
    if isinstance(rB, np.ndarray):
        rB = torch.from_numpy(rB)
    # check if query_label is a numpy array and convert it to a torch tensor if it is
    if isinstance(query_L, np.ndarray):
        query_L = torch.from_numpy(query_L)
    # check if retrieval_label is a numpy array and convert it to a torch tensor if it is
    if isinstance(retrieval_L, np.ndarray):
        retrieval_L = torch.from_numpy(retrieval_L)
    qB, rB, query_L, retrieval_L = [i.cuda().to(dtype=torch.float64) for i in [qB, rB, query_L, retrieval_L]]
    num_query = query_L.shape[0]
    topK = retrieval_L.shape[0]
    query_L = query_L.float().cuda()
    retrieval_L = retrieval_L.float().cuda()
    qB = qB.float().cuda()
    rB = rB.float().cuda()
    GND = (query_L.mm(retrieval_L.t()) > 0).type(torch.float).squeeze()
    # hamm = calc_hamming_dist(qB, rB)
    # _, ind = torch.sort(hamm,dim=1)
    P, R = [], []
    # np.linspace(1,topK+1,20)
    # tqdm(range(num_query))
    for k in tqdm(np.linspace(1,topK+1,15)): # 枚举 top-K 之 K
        k = int(k)
         # ground-truth: 1 vs all
        p = torch.zeros(num_query) # 各 query sample 的 Precision@R
        r = torch.zeros(num_query) # 各 query sample 的 Recall@R
        for it in range(num_query): # 枚举 query sample
            hamm = CalcHammingDist(qB[it, :], rB)
            _, ind = torch.sort(hamm)
            ind=ind.squeeze()
            p[it] = (GND[it][ind[:k]]!=0).sum() / k # 求出所有查询样本的Percision@K
            r[it] = (GND[it][ind[:k]]!=0).sum() / (GND[it]!=0).sum() # 求出所有查询样本的Recall@K
            if (GND[it]!=0).sum() == 0:
                print(1)
        P.append((p.mean()).item())
        R.append((r.mean()).item())
    # return R,P
    return R,P

def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def TCalcTopMap(rB, qB, retrievalL, queryL, topk, tretrievalL, tqueryL):
    num_query = queryL.shape[0]
    topkmap = 0
    temp_ind = 0
    for iter in tqdm(range(num_query)):
        if np.dot(tqueryL[iter,:], queryL[iter,:].transpose()) > 0:
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
            #Cgnd = (np.dot(tqueryL[iter, :], tretrievalL.transpose()) > 0).astype(np.float32)
            hamm = CalcHammingDist(qB[iter, :], rB)
            ind = np.argsort(hamm)
            gnd = gnd[ind]
            #cgnd = Cgnd[ind]

            tgnd = gnd[0:topk]
            #Ntgnd = Ngnd[0:topk]
            tsum = np.sum(tgnd).astype(int)
            if tsum == 0:
                continue
            count = np.linspace(1, tsum, tsum)

            tindex = np.asarray(np.where(tgnd == 1)) + 1.0
            topkmap_ = np.mean(count / (tindex))
            topkmap = topkmap + topkmap_
            temp_ind += 1
    cor_topkmap = topkmap / temp_ind

    topkmap = 0
    temp_ind = 0
    for iter in tqdm(range(num_query)):
        if np.dot(tqueryL[iter,:], queryL[iter,:].transpose()) == 0:
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
            hamm = CalcHammingDist(qB[iter, :], rB)
            ind = np.argsort(hamm)
            gnd = gnd[ind]

            tgnd = gnd[0:topk]
            tsum = np.sum(tgnd).astype(int)
            if tsum == 0:
                continue
            count = np.linspace(1, tsum, tsum)

            tindex = np.asarray(np.where(tgnd == 1)) + 1.0
            topkmap_ = np.mean(count / (tindex))
            topkmap = topkmap + topkmap_
            temp_ind += 1
    oth_topkmap = topkmap / (temp_ind +0.0001)
    return cor_topkmap, oth_topkmap

def plot_gmm(gmm, X, clean_index, noisy_index, save_path='', plot_pdf=True):
    plt.figure(figsize=(7, 5))
    plt.clf()
    ax = plt.gca()

    # Compute PDF of whole mixture
    x = np.linspace(0, 1, 1000)
    logprob = gmm.score_samples(x.reshape(-1, 1))
    pdf = np.exp(logprob)

    # Compute PDF for each component
    responsibilities = gmm.predict_proba(x.reshape(-1, 1))
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    # Plot data histogram
    ax.hist(X[clean_index], bins=100, density=True, histtype='stepfilled', color='green', alpha=0.4, label='Clean Pairs')
    ax.hist(X[noisy_index], bins=100, density=True, histtype='stepfilled', color='red', alpha=0.4, label='Noisy Pairs')

    # Plot PDF of whole model
    if plot_pdf:
        # Plot PDF of each component
        ax.plot(x, pdf_individual[:,  gmm.means_.argmin()], '--', label='Component A', color='green')
        ax.plot(x, pdf_individual[:,  gmm.means_.argmax()], '--', label='Component B', color='red')
        ax.plot(x, pdf, '-k', label='Mixture PDF')
    plt.yticks(size=15)
    plt.xticks(size=15)
    plt.xlabel('Per-sample loss', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.legend(loc='upper right', fontsize=12,frameon=True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    # SaveH5File_I(256)
    SaveH5File_F(256)
    # SaveH5File_C(256)
    # SaveH5File_N(256)
