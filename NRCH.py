from utils.tools import *
import itertools
from scipy.linalg import hadamard
from network import *
import pdb
import os
import torch
import torch.optim as optim
import time
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.mixture import GaussianMixture


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--gpus', type = str, default = '0')
parser.add_argument('--hash_dim', type = int, default = 32)
parser.add_argument('--noise_rate', type = float, default = 0.2)
parser.add_argument('--dataset', type = str, default = 'flickr')
parser.add_argument('--Lambda', type = float, default = 0.6)
parser.add_argument('--num_gradual', type = int, default = 100)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

bit_len = args.hash_dim
noise_rate = args.noise_rate
dataset = args.dataset
Lambda=args.Lambda
num_gradual =  args.num_gradual

if dataset == 'flickr':
    train_size = 10000
elif dataset == 'ms-coco':
    train_size = 10000
elif dataset == 'nuswide21':
    train_size = 10500
elif dataset == 'iapr':
    train_size = 10000
n_class = 0
tag_len = 0
torch.multiprocessing.set_sharing_strategy('file_system')

def get_config():
    config = {
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "txt_optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[CSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size":128,
        "dataset": dataset,
        "epoch": 100,
        "device": torch.device("cuda:0"),
        "bit_len": bit_len,
        "noise_type": 'symmetric',
        "noise_rate": noise_rate,
        "random_state": 1,
        "n_class": n_class,
        "lambda":Lambda,
        "tag_len":tag_len,
        "train_size": train_size,
        "threshold_rate":0.3
    }
    return config    

class Robust_Loss(torch.nn.Module):
    def __init__(self, config, bit):
        super(Robust_Loss, self).__init__()
        self.shift = 1
        self.margin =.2
        self.tau = 1.
    def forward(self, u, v, y,config):
        u = u.tanh()
        v = v.tanh()
        T = self.calc_neighbor (y,y)
        T.diagonal().fill_(0)
        S = u.mm(v.t())
        #pdb.set_trace()
        d = S.diag().view(v.size(0), 1)
        d1 = d.expand_as(S)
        d2 = d.t().expand_as(S)

        mask_te = (S >= (d1 - self.margin)).float().detach()
        cost_te = S * mask_te + (1. - mask_te) * (S - self.shift)

        cost_te_max = torch.zeros_like(cost_te)
        cost_te_max.copy_(cost_te)
        identity_matrix_te = torch.eye(cost_te_max.size(0), cost_te_max.size(1), device=cost_te_max.device, dtype=cost_te_max.dtype)
        diagonal_te = torch.diag(cost_te_max).clamp(min=0)
        modified_diagonal_matrix_te = torch.diag_embed(diagonal_te)
        cost_te_max = cost_te_max * (1 - identity_matrix_te) + modified_diagonal_matrix_te 


        mask_im = (S>= (d2 - self.margin)).float().detach()
        cost_im = S * mask_im + (1. - mask_im) * (S - self.shift)

        cost_im_max = torch.zeros_like(cost_im)
        cost_im_max.copy_(cost_im)
        identity_matrix_im = torch.eye(cost_im_max.size(0), cost_im_max.size(1), device=cost_im_max.device, dtype=cost_im_max.dtype)
        diagonal_im = torch.diag(cost_im_max).clamp(min=0)
        modified_diagonal_matrix_im = torch.diag_embed(diagonal_im)
        cost_im_max = cost_im_max * (1 - identity_matrix_im) + modified_diagonal_matrix_im 
        
        loss_r = (-cost_te.diag()+self.tau * ((cost_te_max / self.tau*(1-T))).exp().sum(1).log() + self.margin) +(-cost_im.diag()+self.tau * ((cost_te_max / self.tau*(1-T))).exp().sum(1).log() + self.margin)
        Q_loss = (u.abs() - 1 / np.sqrt(u.shape[1])).pow(2).mean(axis = 1) + (v.abs() - 1 / np.sqrt(u.shape[1])).pow(2).mean(axis = 1)        
        loss = config["lambda"] *loss_r + (1-config["lambda"])*Q_loss
        final_loss = torch.mean(loss)
        return final_loss
    def calc_neighbor(self,label1, label2):
        # calculate the similar matrix
        label1 = label1.type(torch.cuda.FloatTensor)
        label2 = label2.type(torch.cuda.FloatTensor)
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
        return Sim


def split_prob(prob, threshld):
    pred = (prob >= threshld)
    return (pred+0)

def get_loss(net, txt_net, config, data_loader, Threshold, epoch,W):
    tau = 0.05
    sample_losses = []
    for image, tag, tlabel, label, ind in data_loader:
        image = image.to('cuda')
        image = image.float()
        tag = tag.to('cuda')
        tag = tag.float()
        label = label.to('cuda')
        tlabel = tlabel.to('cuda')
        u = net(image)
        v = txt_net(tag) 
        with torch.no_grad():
            #  add by qy
            label_ = (label - 0.5) * 2  #   [-1,1]
            u_sims = u @ W.tanh().t()   # N X C  [-1, 1]
            v_sims = v @ W.tanh().t()   # N X C  [-1, 1]
            loss_ = (label_ - u_sims)**2
            loss_ += (label_ - v_sims)**2            
            loss = (loss_ * label).max(1)[0]
        right = ((tlabel==label).float().mean(1) == 1).float() 
        for i in range(len(loss)):
            sample_losses.append((ind[i].item(), loss[i].item(), right[i].item()))
    sample_losses_sorted = sorted(sample_losses, key=lambda x: x[0])
    sorted_losses = [item[1] for item in sample_losses_sorted]
    sorted_losses = np.array(sorted_losses)
    sorted_losses = (sorted_losses-sorted_losses.min()+ 1e-8)/(sorted_losses.max()-sorted_losses.min() + 1e-8) 
    sorted_losses = sorted_losses.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=5e-1,reg_covar=5e-4)
    save_path=f'/home/wangla/My_method/final/pdf/loss{epoch}_dataset{config["dataset"]}_bit{config["bit_len"]}_noiserate{config["noise_rate"]}.pdf'
    labels = np.array([item[2] for item in sample_losses_sorted])
    loss = np.array(sorted_losses)
    gmm.fit(sorted_losses)
    prob = gmm.predict_proba(sorted_losses)
    prob = prob[:, gmm.means_.argmin()]
    if epoch+1>=20:
        pred = split_prob(prob,Threshold)
    else:
        pred = split_prob(prob,0)
    clean_index = np.where(labels==1)[0]
    smaller_mean_indices = [i for i, p in enumerate(pred) if p == 1]
    true_positives = set(smaller_mean_indices).intersection(clean_index)
    false_positives = set(smaller_mean_indices).difference(clean_index)
    precision = len(true_positives) / (len(true_positives) + len(false_positives))
    return sorted_losses, torch.Tensor(pred), precision


def train(config, bit):
    device = config["device"]
    train_loader,  test_loader, dataset_loader, num_train,  num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = ImgModule(y_dim=4096, bit=bit, hiden_layer=3).to('cuda')
    txt_net = TxtModule(y_dim=tag_len, bit=bit, hiden_layer=2).to('cuda')
    W = torch.Tensor(n_class, bit_len)
    W = torch.nn.init.orthogonal_(W, gain=1)
    W = torch.tensor(W, requires_grad= True).cuda()
    W = torch.nn.Parameter(W)
    net.register_parameter('W', W) # regist W into the image net
    get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
    params_dnet = get_grad_params(net)
    optimizer = config["optimizer"]["type"](params_dnet, **(config["optimizer"]["optim_params"]))
    txt_optimizer = config["txt_optimizer"]["type"](txt_net.parameters(), **(config["txt_optimizer"]["optim_params"]))
    criterion = Robust_Loss(config, bit)
    threshold_schedule = np.linspace(0, config["threshold_rate"], num_gradual)
    threshold_schedule = np.concatenate((threshold_schedule, np.ones(config["epoch"] - num_gradual) * config["threshold_rate"]))
    i2t_mAP_list = []
    t2i_mAP_list = []
    epoch_list = []
    precision_list = []
    bestt2i=0
    besti2t=0
    n=0
    os.makedirs('./checkpoint', exist_ok=True)
    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")  
        net.eval()
        txt_net.eval()
        net.train()
        txt_net.train()
        train_loss = 0
        if (epoch+1) %20 == 0:
            print("calculating test binary code......")
            img_tst_binary, img_tst_label = compute_img_result(test_loader, net, device=device)
            print("calculating dataset binary code.......")
            img_trn_binary, img_trn_label = compute_img_result(dataset_loader, net, device=device)
            txt_tst_binary, txt_tst_label = compute_tag_result(test_loader, txt_net, device=device)
            txt_trn_binary, txt_trn_label = compute_tag_result(dataset_loader, txt_net, device=device)
            print("calculating map.......")
            t2i_mAP = calc_map_k(img_trn_binary.numpy(), txt_tst_binary.numpy(), img_trn_label.numpy(), txt_tst_label.numpy())
            i2t_mAP = calc_map_k(txt_trn_binary.numpy(),img_tst_binary.numpy(), txt_trn_label.numpy(), img_tst_label.numpy())
            if t2i_mAP+i2t_mAP> bestt2i+besti2t:
                bestt2i=t2i_mAP
                besti2t=i2t_mAP
                torch.save({
                    'net_state_dict': net.state_dict(),
                    'txt_net_state_dict': txt_net.state_dict(),
                }, './checkpoint/best_model.pth') 
            t2i_mAP_list.append(t2i_mAP.item())
            i2t_mAP_list.append(i2t_mAP.item())
            epoch_list.append(epoch)
            print("%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.1f,t2i_mAP:%.3f, i2t_mAP:%.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], config["noise_rate"],t2i_mAP, i2t_mAP))
        sorted_losses, pred, precision = get_loss(net, txt_net, config, train_loader, config["threshold_rate"], epoch, W)
        for image, tag, tlabel, label, ind in train_loader:
            ind_np = ind.cpu().numpy()
            current_pred = pred[ind]
            clean_samples = current_pred == 1
            if clean_samples.sum() > 0:
                image = image[clean_samples].to('cuda')
                image = image.float()
                tag = tag[clean_samples].to('cuda')
                tag = tag.float()
                label = label[clean_samples].to('cuda')
                optimizer.zero_grad()
                txt_optimizer.zero_grad()
                u = net(image)
                v = txt_net(tag) 
                loss = criterion(u, v,label.float(),config)
                label_ = (label - 0.5) * 2  #   [-1,1]
                u_sims = u.detach() @ W.tanh().t()   # N X C  [-1, 1]
                v_sims = v.detach() @ W.tanh().t()   # N X C  [-1, 1]
                loss_ = (label_ - u_sims)**2
                loss_ += (label_ - v_sims)**2
                loss += loss_.mean()
                train_loss += loss
                loss.backward()
                optimizer.step()
                txt_optimizer.step()
        train_loss = train_loss / len(train_loader)
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        precision_list.append(precision)
        print("%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.1f" % (
                config["info"], epoch + 1, bit, config["dataset"], config["noise_rate"]))
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

def test(config, bit, model_path='./checkpoint/best_model.pth'):
    device = config["device"]
    _, test_loader, dataset_loader, _, _, _ = get_data(config)
    net = ImgModule(y_dim=4096, bit=bit, hiden_layer=3).to('cuda')
    txt_net = TxtModule(y_dim=tag_len, bit=bit, hiden_layer=2).to('cuda')
    W = torch.Tensor(n_class, bit_len)
    W = torch.nn.init.orthogonal_(W, gain=1)
    W = torch.tensor(W, requires_grad= True).cuda()
    W = torch.nn.Parameter(W)
    net.register_parameter('W', W)
    # Load the saved models
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net_state_dict'])
    txt_net.load_state_dict(checkpoint['txt_net_state_dict'])
    net.eval()
    txt_net.eval()
    print("calculating test binary code......")
    print("calculating test binary code......")
    img_tst_binary, img_tst_label = compute_img_result(test_loader, net, device=device)
    print("calculating dataset binary code.......")
    img_trn_binary, img_trn_label = compute_img_result(dataset_loader, net, device=device)
    txt_tst_binary, txt_tst_label = compute_tag_result(test_loader, txt_net, device=device)
    txt_trn_binary, txt_trn_label = compute_tag_result(dataset_loader, txt_net, device=device)
    print("calculating map.......")
    t2i_mAP = calc_map_k(img_trn_binary.numpy(), txt_tst_binary.numpy(), img_trn_label.numpy(), txt_tst_label.numpy())
    i2t_mAP = calc_map_k(txt_trn_binary.numpy(),img_tst_binary.numpy(), txt_trn_label.numpy(), img_tst_label.numpy())
    print("Test Results: t2i_mAP: %.3f, i2t_mAP: %.3f" % (t2i_mAP, i2t_mAP))

if __name__ == "__main__":
    data_name_list = ['flickr','nuswide21','iapr','ms-coco']
    bit_list=[64,16,128,32]
    noise_rate_list = [0.2,0.5,0.8]
    for data_name in data_name_list:
        for rate in noise_rate_list:
            for bit in bit_list:
                bit_len = bit
                noise_rate = rate
                dataset = data_name
                if dataset == 'nuswide21':
                    n_class = 21
                    tag_len = 1000
                elif dataset == 'flickr':
                    n_class = 24
                    tag_len = 1386                        
                elif dataset == 'ms-coco':
                    n_class = 80
                    tag_len = 300
                elif dataset == 'iapr':
                    n_class = 255
                    tag_len = 2912
                config = get_config()
                print(config)
                #train(config, bit)
                test(config, bit)