"""
Official implementation for the TMLR paper "FedLGD - Federated Learning on Virtual Heterogeneous Data with Local-GLobal Dataset Distillation"
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
from torch import nn, optim
import time
import copy
import argparse
import numpy as np
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
from pretraineddataset import PretrainedDataset, GetPretrained
from condensation import distribution_matching, get_initial_normal, gradient_matching_all
from torchvision.utils import save_image
import random
from loss_fn import Distance_loss, MMD_loss
import pandas as pd

# import wandb

from fedlgd_data import prepare_data

def train(model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        _, output = model(x)

        loss = loss_fun(output, y)

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data

def train_vhl(model, optimizer, loss_fun, client_num, device, train_loader, server_images, server_labels, distance_loss, lambda_sim, imgs, ipc, server_ipc, reg_loss):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    align_loss_all = 0
    train_iter = iter(train_loader)

    # # get server embedding 
    # server_features = model.embed(server_images)


    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        client_features, output = model(x)

        classification_loss = loss_fun(output, y)

        # similarity model update
        # Constrastive
        if reg_loss == 'contrastive':
            # client_features = model.embed(x)
            server_features = model.embed(server_images)
            align_loss = distance_loss(client_features, server_features, y, server_labels)
        # MMD
        elif reg_loss == 'mmd':
            server_features = model.embed(server_images)
            align_loss = distance_loss(client_features, server_features)
            # for c in range(num_classes):
                # client_img_tmp = imgs[c*ipc:(c+1)*ipc]
                # server_img_tmp = server_images[c*(server_ipc):(c+1)*(server_ipc)]
                # emb_client = model.embed(client_img_tmp)
                # emb_server = model.embed(server_img_tmp)
                # align_loss = torch.sum((torch.mean(emb_server, dim=0) - torch.mean(emb_client, dim=0))**2)
        # l2 norm - will raise error if ipc != server_ipc
        elif reg_loss == 'l2norm':
            for c in range(num_classes):
                client_img_tmp = imgs[c*ipc:(c+1)*ipc]
                server_img_tmp = server_images[c*(server_ipc):(c+1)*(server_ipc)]
                emb_client = model.embed(client_img_tmp)
                emb_server = model.embed(server_img_tmp)
                align_loss = distance_loss(emb_client, emb_server)
        else:
            raise NotImplementedError
        
        # torch.autograd.set_detect_anomaly(True)
        loss = classification_loss + lambda_sim * align_loss

        # loss = classification_loss

        loss.backward(retain_graph=True)
        loss_all += loss.item()
        align_loss_all += align_loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data, align_loss_all/len(train_iter)
    

def train_fedprox(args, model, server_model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        loss = loss_fun(output, y)

        #########################we implement FedProx Here###########################
        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        if step>0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += args.mu / 2. * w_diff
        #############################################################################

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data

def test(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        _, output = model(data)
        
        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()
    
    return test_loss/len(test_loader), correct /len(test_loader.dataset)

################# Key Function ########################
def communication(args, server_model, models, client_weights, client_list):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in client_list: #range(client_num):
                        temp += (1/len(client_list)) * models[client_idx].state_dict()[key]
                        # temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in client_list: #range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in client_list: #range(len(client_weights)):
                        temp += (1/len(client_list)) * models[client_idx].state_dict()[key]
                        # temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in client_list: #range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


def get_images(images_all, indices_class, c, n): # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return images_all[idx_shuffle]




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('Device:', device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help ='whether to make a log')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default= 32, help ='batch size')
    parser.add_argument('--iters', type = int, default=100, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='fedlgd', help='fedavg | fedprox | fedbn | fedlgd')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='./checkpoint', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    
    parser.add_argument('--ipc', type = int, default=50, help = 'images per class')
    parser.add_argument('--server_ipc', type = int, default=10, help = 'server images per class')
    parser.add_argument('--lr_img', type = float, default=1, help = 'learning rate for img')
    parser.add_argument('--dis_metric', type = str, default='ours', help='matching method')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--lambda_sim', type = float, default=1, help = 'lambda for heterogeneous training')
    parser.add_argument('--dataset', type=str, default='digits', help='Dataset: cifar10c | retina | digits')
    parser.add_argument('--seed', type = int, default=0, help = 'random seeds')

    parser.add_argument('--ci_iter', type = int, default=100, help = 'client image update epoch')
    parser.add_argument('--si_iter', type = int, default=500, help = 'server image update epoch')

    parser.add_argument('--ci_tgap', type = int, default=5, help = 'client image training frequency')
    parser.add_argument('--si_tgap', type = int, default=5, help = 'server image training frequency')
    parser.add_argument('--image_update_times', type = int, default=10, help = 'condensed image update times during whole training')

    parser.add_argument('--init', type = str, default='normal', help='initialization method for dc')

    parser.add_argument('--reg_loss', type = str, default='contrastive', help='regularization loss: contrastive|l2norm|mmd')

    parser.add_argument('--save_curves', type = bool, default=False, help='Save loss and acc curves')

    parser.add_argument('--client_ratio', type = float, default=1.0, help = 'client sampling ratio') 

    args = parser.parse_args()
    args.device = device

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)     
    torch.cuda.manual_seed_all(args.seed) 
    random.seed(args.seed)

    assert(args.dataset in ['digits', 'retina', 'cifar10c', 'cifar10c_alpha1', 'cifar10c_alpha5', 'cifar100c'])

    exp_folder = 'federated_' + args.dataset

    args.save_path = os.path.join(args.save_path, args.dataset, exp_folder)

    

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
   
    channel, image_batch = 3, 256
    im_size = (96, 96) if args.dataset == 'retina' else (28, 28)
    if args.dataset == 'retina':
        num_classes = 2
    elif args.dataset == 'cifar100c':
        num_classes = 100
    else:
        num_classes = 10
    # num_classes = 2 if args.dataset == 'retina' else 10
    client_iteration, server_iteration, warmup_iteration = args.ci_iter, args.si_iter, 10000#200, 500, 2000
    # server_model = DigitModel().to(device)
    server_model = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model.to(device)
    # print(server_model)
    # for name, param in server_model.state_dict().items():
    #     print(name)
    # sys.exit()
    loss_fun = nn.CrossEntropyLoss()

    # prepare the data
    train_datasets, test_datasets, train_loaders, test_loaders, concated_test_loader, MEANS, STDS = prepare_data(args, im_size)
    # print([len(trainset) for trainset in test_datasets])
    # sys.exit()
    # print([len(testset) for testset in test_datasets])
    

    # name of each client dataset
    if args.dataset == 'digits':
        datasets = ['MNIST', 'SVHN', 'USPS', 'SynDigits', 'MNIST-M']
    elif args.dataset == 'retina':
        datasets = ['Drishti', 'Kaggle', 'Rim', 'Refuge']
    elif args.dataset == 'cifar10c' or args.dataset == 'cifar10c_alpha1' or args.dataset == 'cifar10c_alpha5':
        datasets = [f'client{i}' for i in range(len(train_datasets))]
    elif args.dataset == 'cifar100c':
        datasets = [f'client{i}' for i in range(len(train_datasets))]

    
    # federated setting
    client_num = len(datasets)
    client_weights = [1/client_num for i in range(client_num)]
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]


    # Save the original data
    image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, device=args.device)
    for client_idx in range(client_num):
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]
        images_all = [torch.unsqueeze(train_datasets[client_idx][i][0], dim=0) for i in range(len(train_datasets[client_idx]))]
        labels_all = [train_datasets[client_idx][i][1] for i in range(len(train_datasets[client_idx]))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        for c in range(num_classes):
            image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(images_all, indices_class, c, args.ipc).detach().data
        
        save_name = os.path.join('result', f'real_image_cifar10c_client{client_idx}.png')
        image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
        for ch in range(channel):
            image_syn_vis[:, ch] = image_syn_vis[:, ch] * STDS[client_idx][ch] + MEANS[client_idx][ch]
        image_syn_vis[image_syn_vis<0] = 0.0
        image_syn_vis[image_syn_vis>1] = 1.0
        save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.
        print(f'client{client_idx} saved')

    sys.exit()

    # make save dictionary
    train_loss_save, train_acc_save, val_loss_save, val_acc_save, test_loss_save, test_acc_save, reg_loss_save, global_loss_save, global_acc_save = {}, {}, {}, {}, {}, {}, {}, {}, {}
    for client_idx in range(client_num):
        train_loss_save[f'Client{client_idx}'] = []
        train_acc_save[f'Client{client_idx}'] = []
        val_loss_save[f'Client{client_idx}'] = []
        val_acc_save[f'Client{client_idx}'] = []
        reg_loss_save[f'Client{client_idx}'] = []
        global_loss_save[f'Client{client_idx}'] = []
        global_acc_save[f'Client{client_idx}'] = []
    train_loss_save[f'mean'] = []
    train_acc_save[f'mean'] = []
    val_loss_save[f'mean'] = []
    val_acc_save[f'mean'] = []
    test_loss_save[f'Global Held Out'] = []
    test_acc_save[f'Global Held Out'] = []
    reg_loss_save[f'mean'] = []
    global_loss_save[f'mean'] = []
    global_acc_save[f'mean'] = []

    if args.test:
        server_model.load_state_dict(torch.load(f'{SAVE_PATH}/server_model_local{args.wk_iters}_{args.iters*args.wk_iters}.pt'))
        # for client_idx in range(client_num):
        #     models[client_idx].load_state_dict(torch.load(f'{SAVE_PATH}/model_local{args.wk_iters}_{client_idx}.pt'))
        
        mean_loss, mean_acc = [], []
        # testing on heldout global test dataset
        for test_idx, test_loader in enumerate(test_loaders):
            test_loss, test_acc = test(server_model, test_loader, loss_fun, device)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
            mean_loss.append(test_loss)
            mean_acc.append(test_acc)
        print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format('Averaged', np.mean(mean_loss), np.mean(mean_acc)))

        # testing on heldout global test dataset
        test_loss, test_acc = test(server_model, concated_test_loader, loss_fun, device)
        print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format('Global Held Out', test_loss, test_acc))

    else:
        # wandb.init(project=f'fedlgd_{args.dataset}')
        # wandb.config = {'wk_iters': args.wk_iters}
        ''' Warm Up: Condense local data before FL'''

        # get initial global and local images
        image_syns_tmp = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syns_tmp = torch.tensor(np.array([np.ones(args.ipc)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        image_syns = [copy.deepcopy(image_syns_tmp).to(args.device) for idx in range(client_num)]
        label_syns = [copy.deepcopy(label_syns_tmp).to(args.device) for idx in range(client_num)]
        server_image_syn = torch.randn(size=(num_classes*args.server_ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        server_label_syn = torch.tensor(np.array([np.ones(args.server_ipc)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        if args.init == 'normal':
            image_syns, server_image_syn = get_initial_normal(train_datasets, im_size, num_classes, client_num, args.ipc, args.server_ipc)
        elif args.init == 'real':
            print('initialize client synthetic data from random real images')
            for client_idx in range(client_num):
                images_all = []
                labels_all = []
                indices_class = [[] for c in range(num_classes)]
                images_all = [torch.unsqueeze(train_datasets[client_idx][i][0], dim=0) for i in range(len(train_datasets[client_idx]))]
                labels_all = [train_datasets[client_idx][i][1] for i in range(len(train_datasets[client_idx]))]
                for i, lab in enumerate(labels_all):
                    indices_class[lab].append(i)
                images_all = torch.cat(images_all, dim=0).to(args.device)
                labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
                for c in range(num_classes):
                    image_syns[client_idx].data[c*args.ipc:(c+1)*args.ipc] = get_images(images_all, indices_class, c, args.ipc).detach().data
        # else: 
        #     raise NotImplementedError


        ''' Pre-processing clients' data '''
        # from pre-trained
        pretrained_img_path = f'./pretrained/{args.dataset}/'
        if os.path.isfile(f'{pretrained_img_path}/{args.model}_{args.ipc}_{args.init}_client0_iter0_local1.png'):
            path_tmp = f'{pretrained_img_path}/{args.model}_{args.ipc}_{args.init}'
            image_syns = GetPretrained(path=path_tmp, means=MEANS, stds=STDS, im_size=im_size, num_classes=num_classes, client_num=client_num, device=args.device, ipc = args.ipc, padding = 2)
            for i, local_syn_images in enumerate(image_syns):
                # for ch in range(channel):
                #     local_syn_images[:, ch] = (local_syn_images[:, ch] - MEANS[i][ch]) /STDS[i][ch]
                local_syn_images.requires_grad = True
        # local DM 
        else:
            for client_idx in range(client_num):
                # organize the real dataset
                images_all = []
                labels_all = []
                indices_class = [[] for c in range(num_classes)]
                images_all = [torch.unsqueeze(train_datasets[client_idx][i][0], dim=0) for i in range(len(train_datasets[client_idx]))]
                labels_all = [train_datasets[client_idx][i][1] for i in range(len(train_datasets[client_idx]))]
                for i, lab in enumerate(labels_all):
                    indices_class[lab].append(i)
                images_all = torch.cat(images_all, dim=0).to(args.device)
                labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
                # print(client_idx, [len(indices_class[idx]) for idx in range(num_classes)])
                
                # setup optimizer
                optimizer_img = torch.optim.SGD([image_syns[client_idx], ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
                optimizer_img.zero_grad()
                
                for it in range(warmup_iteration):
                    loss_avg = 0
                    # get real images for each class
                    image_real = [get_images(images_all, indices_class, c, image_batch) for c in range(num_classes)]
                    if 'BN' in args.model:
                        loss, image_syns[client_idx] = distribution_matching_bn(image_real, image_syns[client_idx], optimizer_img, channel, num_classes, im_size, args.ipc)
                    else:
                        loss, image_syns[client_idx] = distribution_matching(image_real, image_syns[client_idx], optimizer_img, channel, num_classes, im_size, args.ipc)
                    # report averaged loss
                    loss_avg += loss
                    loss_avg /= num_classes
                    if it%100 == 0:
                        print('%s Initialization:\t client = %2d, iter = %05d, loss = %.4f' % (get_time(), client_idx, it, loss_avg))
        # sys.exit()
        
        # args.lr_img = args.lr_img/2
        

        ''' start training with condensed images '''
        local_time = 0
        global_time = 0
        start_time = time.time()
        for a_iter in range(0, args.iters+1):

            client_list = np.random.choice(np.arange(client_num), int(args.client_ratio*client_num), replace=False)
            print(F'Selected {int(args.client_ratio*client_num)} clients for round {a_iter}:')
            print(client_list)

            # # slow down lr for gradient matching
            # if a_iter < args.si_tgap*args.image_update_times+1:
            #     model_lr = args.lr/10
            # else:
            #     model_lr = args.lr
            model_lr = args.lr

            # Save distilled data for future initialization
            if ((a_iter+1)%args.si_tgap == 0 or a_iter == 0) and a_iter < args.si_tgap*args.image_update_times+1:
                # save local distilled data
                data_path = f'{pretrained_img_path}'
                if not os.path.exists(data_path):
                    os.makedirs(data_path)
                for i, local_syn_images in enumerate(image_syns):
                    save_name = os.path.join(data_path, f'{args.model}_{args.ipc}_{args.init}_client{i}_iter{a_iter}_local{args.wk_iters}.png')
                    image_syn_vis = copy.deepcopy(local_syn_images.detach().cpu())
                    for ch in range(channel):
                        image_syn_vis[:, ch] = image_syn_vis[:, ch] * STDS[i][ch] + MEANS[i][ch]
                    image_syn_vis[image_syn_vis<0] = 0.0
                    image_syn_vis[image_syn_vis>1] = 1.0
                    save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.


            if ((a_iter+1)%args.si_tgap == 0 or a_iter == 0) and a_iter < args.si_tgap*args.image_update_times+1:
            # if a_iter==199:
                # save global distilled data
                data_path = f'{SAVE_PATH}/distilled_data'
                if not os.path.exists(data_path):
                    os.makedirs(data_path)
                save_name = os.path.join(data_path, f'{args.model}_{args.ipc}_{args.init}_global_v2_iter{a_iter}_local{args.wk_iters}.png')
                image_syn_vis = copy.deepcopy(server_image_syn.detach().cpu())
                mean_global = np.mean(MEANS, axis=0)
                std_global = np.mean(STDS, axis=0)
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std_global[ch] + mean_global[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.server_ipc) # Trying normalize = True/False may get better visual effects.

            # if ((a_iter+1)%args.ci_tgap == 0 or a_iter == 0) and a_iter < args.ci_tgap*args.image_update_times+1:
            #     # save local distilled data
            #     for i, local_syn_images in enumerate(image_syns):
            #         save_name = os.path.join(data_path, f'{args.model}_{args.ipc}_{args.init}_client{i}_iter{a_iter}_local{args.wk_iters}.png')
            #         image_syn_vis = copy.deepcopy(local_syn_images.detach().cpu())
            #         for ch in range(channel):
            #             image_syn_vis[:, ch] = image_syn_vis[:, ch] * STDS[i][ch] + MEANS[i][ch]
            #         image_syn_vis[image_syn_vis<0] = 0.0
            #         image_syn_vis[image_syn_vis>1] = 1.0
            #         save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

            if a_iter == args.iters:
                break

            
            ''' Update local condensed data with DM '''
            # if a_iter > 0 and a_iter < 11:
            # if a_iter > 0 and a_iter%10==0:
            # if a_iter > 0 and a_iter%10==0 and a_iter<101:
            if a_iter > 0 and a_iter%args.ci_tgap==0 and a_iter < args.ci_tgap*args.image_update_times+1:
                tstart = time.time()
                for client_idx in client_list: #range(client_num):
                    # organize the real dataset - can move outside the loop
                    images_all = []
                    labels_all = []
                    indices_class = [[] for c in range(num_classes)]
                    images_all = [torch.unsqueeze(train_datasets[client_idx][i][0], dim=0) for i in range(len(train_datasets[client_idx]))]
                    labels_all = [train_datasets[client_idx][i][1] for i in range(len(train_datasets[client_idx]))]
                    for i, lab in enumerate(labels_all):
                        indices_class[lab].append(i)
                    images_all = torch.cat(images_all, dim=0).to(args.device)
                    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
                    
                    # setup optimizer
                    optimizer_img = torch.optim.SGD([image_syns[client_idx], ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
                    optimizer_img.zero_grad()
                    # get global condensed images as ref
                    # image_server = [copy.deepcopy(server_image_syn[c*(args.ipc):(c+1)*(args.ipc)].detach()).to(args.device) for c in range(num_classes)]
                    for it in range(client_iteration):
                        loss_avg = 0
                        # get real images for each class
                        image_real = [get_images(images_all, indices_class, c, image_batch) for c in range(num_classes)]
                        # loss, image_syns[client_idx], sc_loss = distribution_matching(image_real, image_syns[client_idx], optimizer_img, channel, num_classes, im_size, args.ipc, image_server=image_server)
                        if 'BN' in args.model:
                            loss, image_syns[client_idx] = distribution_matching_bn(image_real, image_syns[client_idx], optimizer_img, channel, num_classes, im_size, args.ipc, net=server_model)
                        else:
                            loss, image_syns[client_idx] = distribution_matching(image_real, image_syns[client_idx], optimizer_img, channel, num_classes, im_size, args.ipc, net=server_model)
                        # report averaged loss
                        loss_avg += loss
                        loss_avg /= num_classes
                        # if it == iteration-1:
                        if (it+1)%100==0:
                            print('%s Local update:\t client = %2d, Total iter:%05d, iter = %05d, loss = %.4f' % (get_time(), client_idx, a_iter, it, loss_avg))
                local_time += time.time() - tstart
                print(f'Local synthetic images update time per iteration: {time.time() - tstart}')


            ''' Ordinary local training with condensed images '''
            tstart = time.time()
            optimizers = [optim.SGD(params=models[idx].parameters(), lr=model_lr) for idx in range(client_num)]
            if args.reg_loss == 'contrastive':
                distance_loss = Distance_loss(device=args.device)
            elif args.reg_loss == 'l2norm':
                distance_loss = nn.MSELoss()
            elif args.reg_loss == 'mmd':
                distance_loss = MMD_loss()
            else:
                raise NotImplementedError

            # deep copy for any unawared modification
            image_server_tmp = copy.deepcopy(server_image_syn.detach().to(args.device))
            label_server_tmp = copy.deepcopy(server_label_syn.detach().to(args.device))
            # dst_server = TensorDataset(image_server_tmp, label_server_tmp)
            # ldr_server = torch.utils.data.DataLoader(dst_server, batch_size=len(dst_server), shuffle=True, num_workers=0)

            image_syn_evals = [copy.deepcopy(image_syns[idx].detach()).to(args.device) for idx in range(client_num)]
            label_syn_evals = [copy.deepcopy(label_syns[idx].detach()).to(args.device) for idx in range(client_num)]
            # dst_clients = [TensorDataset(image_syn_evals[idx], label_syn_evals[idx]) for idx in range(client_num)]
            # ldr_clients = [torch.utils.data.DataLoader(dst_clients[idx], batch_size=len(dst_clients), shuffle=True, num_workers=0) for idx in range(client_num)]

            dst_trains = [TensorDataset(torch.cat([image_syn_evals[idx], image_server_tmp], dim=0), torch.cat([label_syn_evals[idx], label_server_tmp], dim=0)) for idx in range(client_num)]
            ldr_trains = [torch.utils.data.DataLoader(dst_trains[idx], batch_size=args.batch, shuffle=True, num_workers=0) for idx in range(client_num)]

            if  a_iter%args.si_tgap==0 and a_iter < args.si_tgap*args.image_update_times+1:
                for wi in range(args.wk_iters):
                    print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
                    # train local model using local condensed images
                    mean_loss = []
                    for client_idx in client_list: #range(client_num):
                        model, optimizer, ldr_train, image_syn_eval = models[client_idx], optimizers[client_idx], ldr_trains[client_idx], image_syn_evals[client_idx]
                        _, _ = train(model, ldr_train, optimizer, loss_fun, client_num, device)
                        # record align_loss
                        # wandb.log({f'Align Loss Client {client_idx}': 0})
                        reg_loss_save[f'Client{client_idx}'].append(0)
                        mean_loss.append(0)
                        if client_idx == client_num-1:
                            reg_loss_save['mean'].append(np.mean(mean_loss))
                            print(f'Avg align loss = {np.mean(mean_loss)}')
            else:
                for wi in range(args.wk_iters):
                    print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
                    # train local model using local condensed images
                    mean_loss = []
                    for client_idx in client_list: #range(client_num):
                        model, optimizer, ldr_train, image_syn_eval = models[client_idx], optimizers[client_idx], ldr_trains[client_idx], image_syn_evals[client_idx]
                        _, _, align_loss = train_vhl(model, optimizer, loss_fun, client_num, device, ldr_train, image_server_tmp, label_server_tmp, distance_loss, args.lambda_sim, image_syn_eval, args.ipc, args.server_ipc, args.reg_loss)
                        # record align_loss
                        # wandb.log({f'Align Loss Client {client_idx}': align_loss})
                        reg_loss_save[f'Client{client_idx}'].append(align_loss)
                        mean_loss.append(align_loss)
                        if client_idx == client_num-1:
                            reg_loss_save['mean'].append(np.mean(mean_loss))
                            print(f'Avg align loss = {np.mean(mean_loss)}')
            local_time += time.time() - tstart
            print(f'FL pipeline time per iteration: {time.time() - tstart}')


            # make a copy of the server model weights before aggregation
            server_model_old = copy.deepcopy(server_model)


            ''' Aggregation '''
            tstart = time.time()
            server_model, models = communication(args, server_model, models, client_weights, client_list)
            global_time += time.time() - tstart
            if (wi + a_iter * args.wk_iters + 1)%50 == 0:
                torch.save(server_model.state_dict(), f'{SAVE_PATH}/server_model_local{args.wk_iters}_{wi + a_iter * args.wk_iters + 1}.pt')

            

            ''' Update global condensed data with GM '''
            # if a_iter < 11:
            # if a_iter%10==0 and a_iter<101:
            # if a_iter%5==0 and a_iter<51:
            # if a_iter<51:
            if  a_iter%args.si_tgap==0 and a_iter < args.si_tgap*args.image_update_times+1:
                tstart = time.time()
                
                # # setup optimizer and criterion
                optimizer_img = torch.optim.SGD([server_image_syn,], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
                optimizer_img.zero_grad()
                criterion = nn.CrossEntropyLoss().to(args.device)
                
                # # set dummy gradient template
                # gw_reals_ = []
                # local_datas = [copy.deepcopy(image_syns[idx].detach()).to(args.device) for idx in range(client_num)]
                # local_labels = [copy.deepcopy(label_syns[idx].detach()).to(args.device) for idx in range(client_num)]
                # for client_idx in range(client_num):
                #     local_data = local_datas[client_idx].reshape((args.ipc*num_classes, channel, im_size[0], im_size[1]))
                #     local_label = local_labels[client_idx]
                #     _, output_local = server_model_old(local_data)
                #     loss_local = criterion(output_local, local_label)
                #     if gw_reals_ == []:
                #         gw_real_ = torch.autograd.grad(loss_local.to(args.device), list(server_model_old.parameters()))
                #         gw_reals_ = list((ggw.detach().clone()*client_weights[client_idx] for ggw in gw_real_))
                #     else:
                #         gw_real_ = torch.autograd.grad(loss_local.to(args.device), list(server_model_old.parameters()))
                #         gw_real_ = list((ggw.detach().clone()*client_weights[client_idx] for ggw in gw_real_))
                #         for i, gw in enumerate(gw_real_):
                #             gw_reals_[i] += gw


                # calculate the updating gradients
                # pp_tmp_new = []
                # for i, p in enumerate(server_model.parameters()):
                #     pp_tmp_new.append(p.data)
                # pp_tmp_old = []
                # for i, p in enumerate(server_model_old.parameters()):
                #     pp_tmp_old.append(copy.deepcopy(p.data.detach()).to(args.device))

                gw_reals = []
                for p_new, p_old in zip(server_model.parameters(), server_model_old.parameters()):
                    gw_reals.append((p_old.data - p_new.data)/args.lr)
                # print(gw_reals[-1])
                # print(gw_reals_[-1])
                # for ggg, ggg_ in zip(gw_reals, gw_reals_):
                #     print(np.sum((ggg-ggg_).detach().cpu().numpy()))

                # print(gw_reals[-2])
                # print(gw_reals_[-2])
                # sys.exit()
                
                local_time += time.time() - tstart

                # gradient matching
                tstart = time.time()
                for it in range(server_iteration):
                    # loss = gradient_inversion(args, server_model, criterion, optimizer_img, server_image_syn, server_label_syn, num_classes, args.server_ipc, channel, im_size)
                    loss = gradient_matching_all(args, server_model_old, criterion, gw_reals, server_image_syn, server_label_syn, optimizer_img, channel, num_classes, im_size, args.server_ipc)
                    # loss, server_image_syn = gradient_distribution_matching(args, server_model, criterion, gw_reals, local_datas, server_image_syn, optimizer_img, channel, num_classes, im_size, args.ipc//5)
                    # report averaged loss
                    loss /= num_classes
                    # wandb.log({f'Gradient Inversion Loss': loss})
                    # if it == server_iteration-1:
                    if (it+1)%100==0:
                        print('Global update:\t Total iter:%05d, local iter = %05d, loss = %.4f' % (a_iter, it, loss))
                global_time += time.time()-tstart
                print(f'Global synthetic images update time per iteration: {time.time()-tstart}')

            

            ''' Report after aggregation '''
            # mean_loss, mean_acc = [], []
            # for client_idx in client_list: #range(client_num):
            #     model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
            #     train_loss, train_acc = test(server_model, train_loader, loss_fun, device) 
            #     print(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx] ,train_loss, train_acc))
            #     train_loss_save[f'Client{client_idx}'].append(train_loss)
            #     train_acc_save[f'Client{client_idx}'].append(train_loss)
            #     mean_loss.append(train_loss)
            #     mean_acc.append(train_acc)
            #     if client_idx == client_num-1:
            #         train_loss_save['mean'].append(np.mean(mean_loss))
            #         train_acc_save['mean'].append(np.mean(mean_acc))
                    
            # testing
            mean_loss, mean_acc = [], []
            for test_idx, test_loader in enumerate(test_loaders):
                test_loss, test_acc = test(server_model, test_loader, loss_fun, device)
                print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx], test_loss, test_acc))
                # wandb.log({f'Test Loss client {test_idx}': test_loss})
                # wandb.log({f'Test Acc client {test_idx}': test_acc})
                val_loss_save[f'Client{test_idx}'].append(test_loss)
                val_acc_save[f'Client{test_idx}'].append(test_acc)
                mean_loss.append(test_loss)
                mean_acc.append(test_acc)
                if test_idx == client_num-1:
                    val_loss_save['mean'].append(np.mean(mean_loss))
                    val_acc_save['mean'].append(np.mean(mean_acc))
            
            print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format('Averaged', val_loss_save['mean'][-1], val_acc_save['mean'][-1]))
            # wandb.log({f'Averaged Test Acc': val_acc_save['mean'][-1]})

            # testing on heldout global test dataset
            test_loss, test_acc = test(server_model, concated_test_loader, loss_fun, device)
            print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format('Global Held Out', test_loss, test_acc))
            # wandb.log({'Test Loss global held-out': test_loss})
            # wandb.log({'Test Acc global held-out': test_acc})
            test_loss_save['Global Held Out'].append(test_loss)
            test_acc_save['Global Held Out'].append(test_acc)
            
                
        print(f'Total elapsed time: {time.time()-start_time} secs')
        # wandb.log({'Total elapsed time': (time.time()-start_time)/60})
        print(f'Total server time: {global_time} secs')
        print(f'Total client time: {local_time} secs')


        ''' Use globally distilled data for training '''

        # deep copy for any unawared modification
        server_image_final = copy.deepcopy(server_image_syn.detach()).to(args.device)
        server_label_final = copy.deepcopy(server_label_syn.detach()).to(args.device)
        server_dst_train_final = TensorDataset(server_image_final, server_label_final)
        server_loader_final = torch.utils.data.DataLoader(server_dst_train_final, batch_size=args.server_ipc, shuffle=True, num_workers=0)
        
        # get model and set criterion and optimizer
        server_model_final = get_network(args.model, channel, num_classes, im_size).to(args.device)
        loss_fun_final = nn.CrossEntropyLoss()
        optimizer_final = optim.SGD(params=server_model_final.parameters(), lr=args.lr)
        for wi in range(args.iters*args.wk_iters):
            # train global model using global condensed images
            train(server_model_final, server_loader_final, optimizer_final, loss_fun_final, client_num, args.device)
        
        # testing on local datasets
        mean_loss, mean_acc = [], []
        for test_idx, test_loader in enumerate(test_loaders):
            test_loss, test_acc = test(server_model_final, test_loader, loss_fun_final, args.device)
            # wandb.log({f'Final Global Test Acc Client {test_idx}': test_acc})
            print('Final global model {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx], test_loss, test_acc))
            global_loss_save[f'Client{test_idx}'].append(test_loss)
            global_acc_save[f'Client{test_idx}'].append(test_acc)
            mean_loss.append(test_loss)
            mean_acc.append(test_acc)
            if test_idx == client_num-1:
                global_loss_save['mean'].append(np.mean(mean_loss))
                global_acc_save['mean'].append(np.mean(mean_acc))
            

        ''' Save checkpoint '''
        print(' Saving checkpoints to {}...'.format(SAVE_PATH))
        for client_idx in range(client_num):
            torch.save(models[client_idx].state_dict(), f'{SAVE_PATH}/model_local{args.wk_iters}_{client_idx}.pt')
        torch.save(server_model.state_dict(), f'{SAVE_PATH}/server_model_local{args.wk_iters}_{args.iters*args.wk_iters}.pt')

        # # Save acc and loss results
        # metrics_pd = pd.DataFrame.from_dict(train_loss_save)
        # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"fedlgd_{args.model}_{args.dataset}_train_loss_IPC{args.ipc}_{args.seed}.csv"))
        # metrics_pd = pd.DataFrame.from_dict(train_acc_save)
        # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"{args.model}_{args.ipc}_{args.init}_{args.reg_loss}{args.lambda_sim}_train_acc_local{args.wk_iters}_{args.seed}.csv"))
        # metrics_pd = pd.DataFrame.from_dict(val_loss_save)
        # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"fedlgd_{args.model}_{args.dataset}_val_loss_IPC{args.ipc}_{args.seed}.csv"))
        # metrics_pd = pd.DataFrame.from_dict(val_acc_save)
        # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"fedlgd_{args.dataset}_val_acc_{args.seed}.csv"))
        # metrics_pd = pd.DataFrame.from_dict(test_loss_save)
        # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"{args.model}_{args.ipc}_{args.init}_{args.reg_loss}{args.lambda_sim}_test_loss_local{args.wk_iters}_{args.seed}.csv"))
        # metrics_pd = pd.DataFrame.from_dict(test_acc_save)
        # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"{args.model}_{args.ipc}_{args.init}_{args.reg_loss}{args.lambda_sim}_test_acc_local{args.wk_iters}_{args.seed}.csv"))

        # metrics_pd = pd.DataFrame.from_dict(reg_loss_save)
        # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"{args.model}_{args.ipc}_{args.init}_{args.reg_loss}{args.lambda_sim}_local{args.wk_iters}_{args.seed}.csv"))

        # metrics_pd = pd.DataFrame.from_dict(global_loss_save)
        # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"{args.model}_{args.ipc}_{args.init}_{args.reg_loss}{args.lambda_sim}_global_loss_local{args.wk_iters}_{args.seed}.csv"))
        # metrics_pd = pd.DataFrame.from_dict(global_acc_save)
        # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"{args.model}_{args.ipc}_{args.init}_{args.reg_loss}{args.lambda_sim}_global_acc_local{args.wk_iters}_{args.seed}.csv"))

        

