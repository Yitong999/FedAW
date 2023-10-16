import os
import pickle
from tqdm import tqdm
from datetime import datetime
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import math
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

from config import ex
from data.util import get_dataset, IdxDataset, ZippedDataset, average_weights, DatasetSplit
from module.loss import GeneralizedCELoss
from module.util import get_model
from util import MultiDimAverageMeter, EMA
from data.sampling import noniid, iid, extreme_noniid
import random

seed=0
torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# BiasAdv
def pgd_attack_adv(device, model_b, model_d, images, labels, eps=0.4, alpha=4/255, lmd = 2, iters=40) :
    images = images.to(device)
    labels = labels.to(device)

    loss = nn.CrossEntropyLoss(reduction='none')
        
    ori_images = images.data
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs_b = model_b(images)
        outputs_d = model_d(images)

        model_b.zero_grad()
        model_d.zero_grad()
        
        cost_b = loss(outputs_b, labels).to(device)
        cost_d = loss(outputs_d, labels).to(device)
        
        cost = (cost_b - lmd * cost_d).mean()
        
        cost.backward()

        adv_images = images + alpha*images.grad.sign()

        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    
    mode = 3
    
    if mode == 0:
        print('###################')
        print('label: ', labels)
        print('ori predicted by biased model: ', model_b(ori_images))
        print('ori predicted by debiased model: ', model_d(ori_images))
        print('adv predicted by biased model: ', model_b(images))
        print('adv predicted by debiased model: ', model_d(images))
    elif mode == 1:
        print('###################')
        print('label: ', labels)
        print('ori predicted by biased model: ', torch.argmax(model_b(ori_images), dim=1))
        print('ori predicted by debiased model: ', torch.argmax(model_d(ori_images), dim=1))
        print('adv predicted by biased model: ', torch.argmax(model_b(images), dim=1))
        print('adv predicted by debiased model: ', torch.argmax(model_d(images), dim=1))
    else:
        return images
    
        
    return images

def pgd_attack_both_adv(device, model_global, model_b, model_d, images, labels, eps=0.4, alpha=4/255, lmd = 1, iters=40) :
    images = images.to(device)
    labels = labels.to(device)

    loss = nn.CrossEntropyLoss(reduction='none')
        
    ori_images = images.data
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs_b = model_b(images)
        outputs_d = model_d(images)
        outputs_global = model_global(images)
        # outputs1 = model(images)
        # output2
        model_b.zero_grad()
        model_d.zero_grad()
        model_global.zero_grad()
        
        # print('###shape output###: ', outputs_b.shape)
        # print('###shape labels###: ', labels.shape)
        cost_b = loss(outputs_b, labels).to(device)
        

        cost_d = loss(outputs_d, labels).to(device)
        
        cost_global = loss(outputs_global, labels).to(device)

        cost = (cost_b - lmd * cost_global - lmd * cost_d).mean()
#         cost = (cost_b - lmd * cost_d).mean()
        
        
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
#         adv_images = images
        
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    
    mode = 3
    
    if mode == 0:
        print('###################')
        print('label: ', labels)
        print('ori predicted by biased model: ', model_b(ori_images))
        print('ori predicted by debiased model: ', model_d(ori_images))
        print('adv predicted by biased model: ', model_b(images))
        print('adv predicted by debiased model: ', model_d(images))
    elif mode == 1:
        print('###################')
        print('label: ', labels)
        print('ori predicted by biased model: ', torch.argmax(model_b(ori_images), dim=1))
        print('ori predicted by debiased model: ', torch.argmax(model_d(ori_images), dim=1))
        print('adv predicted by biased model: ', torch.argmax(model_b(images), dim=1))
        print('adv predicted by debiased model: ', torch.argmax(model_d(images), dim=1))
    else:
        return images
    
        
    return images


@ex.automain
def train(
    main_tag,
    dataset_tag,
    model_tag,
    data_dir,
    log_dir,
    device,
    target_attr_idx,
    bias_attr_idx,
    main_num_steps,
    main_valid_freq,
    main_batch_size,
    main_optimizer_tag,
    main_learning_rate,
    main_weight_decay,
):

    print(dataset_tag) #ColoredMNIST

    device = torch.device(device)
    start_time = datetime.now()
    writer = SummaryWriter(os.path.join(log_dir, "summary", main_tag))


    
    train_dataset = get_dataset(
            dataset_tag,
            data_dir=data_dir,
            dataset_split="train",
            transform_split="train",
        )
    
    data_set_tag_list = ['ColoredMNIST-Skewed0.001-Severity3', 'ColoredMNIST-Skewed0.001-Severity4', 
                         'ColoredMNIST-Skewed0.005-Severity3', 'ColoredMNIST-Skewed0.005-Severity4',
                         'ColoredMNIST-Skewed0.01-Severity3',  'ColoredMNIST-Skewed0.01-Severity4',
                         'ColoredMNIST-Skewed0.02-Severity3',  'ColoredMNIST-Skewed0.02-Severity4',
                         'ColoredMNIST-Skewed0.05-Severity3', 'ColoredMNIST-Skewed0.05-Severity4']
   
    
    
    train_loader_list = []
    valid_loader_list = []

    for tag in data_set_tag_list:
        print('tag in each: ', tag)
        print('data_dir: ', data_dir)
        train_dataset = get_dataset(
            dataset_tag = tag,
            data_dir=data_dir,
            dataset_split="train",
            transform_split="train",
        )

        valid_dataset = get_dataset(
                dataset_tag = tag,
                data_dir=data_dir,
                dataset_split="eval",
                transform_split="eval",
            )

        train_dataset = IdxDataset(train_dataset)
        valid_dataset = IdxDataset(valid_dataset)  

        # make loader    
        train_loader = DataLoader(
            train_dataset,
            batch_size=512,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=1000,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        train_loader_list.append(train_loader)
        valid_loader_list.append(valid_loader)


        # check
        # train_iter = iter(train_loader)
        # index, data, attr = next(train_iter)

        # print(index.shape, data.shape, attr.shape)

        # data = data.to(device)
        # attr = attr.to(device)
        # label = attr[:, target_attr_idx]
        # color = attr[:, bias_attr_idx]
        
        # # Count the number of elements with label == color
        # check = torch.sum(label == color).item()

        # print(f'[check in inilialization]number of bias aligned samples: {check}')
        # print(f'[check in initialization]number of bias conflicting samples: {len(label) - check}')

    '''
     # train_dataset = train_dataset_list[1]
    train_dataset = get_dataset(
            dataset_tag,
            data_dir=data_dir,
            dataset_split="train",
            transform_split="train",
        )
    
    num_users = 10
    frac = 1

    user_groups = iid(train_dataset, num_users)
    # print('user_groups: ', user_groups)
    train_dataset_list = [
        DatasetSplit(train_dataset, user_groups[i])
        for i in range(10)]
    
    
    
    valid_dataset = get_dataset(
        dataset_tag,
        data_dir=data_dir,
        dataset_split="eval",
        transform_split="eval",
    )

    
    # print('user_groups: ', user_groups)

    train_target_attr = train_dataset.attr[:, target_attr_idx]
    train_bias_attr = train_dataset.attr[:, bias_attr_idx]

    attr_dims = []

    attr_dims.append(torch.max(train_target_attr).item() + 1)
    attr_dims.append(torch.max(train_bias_attr).item() + 1)

    num_classes = attr_dims[0]

    # train_dataset = IdxDataset(train_dataset)

    # for i in range(10):
    #     # train_dataset_list[i] = DatasetSplit(train_dataset, user_groups[i])
    #     train_dataset_list[i] = DatasetSplit(train_dataset)
    
    valid_dataset = IdxDataset(valid_dataset)
    
# make loader
    train_loader_list = [
        DataLoader(
            train_dataset_list[i],
            batch_size=main_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        ) 
    for i in range(10)]

    train_loader = DataLoader(
        train_dataset,
        # train_dataset_list[0],
        batch_size=main_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1000,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    
    
    '''
#     train_target_attr = train_dataset.attr[:, target_attr_idx]
#     train_bias_attr = train_dataset.attr[:, bias_attr_idx]

#     attr_dims = []

#     attr_dims.append(torch.max(train_target_attr).item() + 1)
#     attr_dims.append(torch.max(train_bias_attr).item() + 1)


    attr_dims = [10, 10]
    num_classes = attr_dims[0]

    model_global = get_model(model_tag, num_classes).to(device)
    # model_global_initial = get_model(model_tag, attr_dims[0]).to(device)

    model_biased = get_model(model_tag, num_classes).to(device)

    # Training
    def update_weights(model_b, model_d, client, epochs, local_epochs=10):
        # Set mode to train model
        model_b.train()
        model_d.train()
        epoch_loss = []
        
        
        if main_optimizer_tag == "SGD":
            optimizer_b = torch.optim.SGD(
                model_b.parameters(),
                lr=main_learning_rate,
                weight_decay=main_weight_decay,
                momentum=0.9,
            )
            optimizer_d = torch.optim.SGD(
                model_d.parameters(),
                lr=main_learning_rate,
                weight_decay=main_weight_decay,
                momentum=0.9,
            )

            # scheduler_b = torch.optim.lr_scheduler.StepLR(optimizer_b, step_size=1, gamma=0.1)
            # scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=1, gamma=0.1)

        elif main_optimizer_tag == "Adam":
            optimizer_b = torch.optim.Adam(
                model_b.parameters(),
                lr=main_learning_rate,
                weight_decay=main_weight_decay,
            )
            optimizer_d = torch.optim.Adam(
                model_d.parameters(),
                lr=main_learning_rate,
                weight_decay=main_weight_decay,
            )
        elif main_optimizer_tag == "AdamW":
            optimizer_b = torch.optim.AdamW(
                model_b.parameters(),
                lr=main_learning_rate,
                weight_decay=main_weight_decay,
            )
            optimizer_d = torch.optim.AdamW(
                model_d.parameters(),
                lr=main_learning_rate,
                weight_decay=main_weight_decay,
            )
        else:
            raise NotImplementedError
        
        # define loss
        criterion = nn.CrossEntropyLoss(reduction='none')
        bias_criterion = GeneralizedCELoss() #hacked

        
        # sample_loss_ema_b = EMA(torch.LongTensor(train_target_attr), alpha=0.7)
        # sample_loss_ema_d = EMA(torch.LongTensor(train_target_attr), alpha=0.7)
        

        for step in tqdm(range(local_epochs)):
            batch_loss = []
            batch_loss_ori = []
            batch_loss_adv = []

            # train main model
            try:
                index, data, attr = next(train_iter)
            except:
               
                loader = train_loader_list[client]
                print('### client: ', client)
                train_iter = iter(loader)
                
                index, data, attr = next(train_iter)
                
            #original version
            data = data.to(device)
            attr = attr.to(device)
            label = attr[:, target_attr_idx]
            color = attr[:, bias_attr_idx]
            
            # Count the number of elements with label == color
            
            # check = torch.sum(label == color).item()

            # print(f'number of bias aligned samples: {check}')
            # print(f'number of bias conflicting samples: {len(label) - check}')
            

            #hacked verison
            data_adv = pgd_attack_adv(device, model_b, model_d, data, label)
            # data_adv = pgd_attack_both_adv(device, model_global, model_b, model_global, data, label)
    

            # count = [0 for i in range(10)]
            # for j in range(512):
            #     count[label[j]] += 1

            # print(f'client[{client}] images label: ', count)

            
            
            logit_b = model_b(data)
            if np.isnan(logit_b.mean().item()):
                print(logit_b)
                raise NameError('logit_b')
            logit_d = model_d(data)
            logit_d_adv = model_d(data_adv) 
            
            loss_b = criterion(logit_b, label).cpu().detach()
            loss_d = criterion(logit_d, label).cpu().detach()
            
            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d')
            
            loss_per_sample_b = loss_b
            loss_per_sample_d = loss_d

            loss_weight = loss_b / (loss_b + loss_d + 1e-8)

            beta = 0.4
                
            loss_weight_adv = beta * (1 - loss_weight)
            
            if np.isnan(loss_weight.mean().item()):
                print('loss_weight mean: ', loss_weight.mean())
                print('loss_weight mean item: ', loss_weight.mean().item())
                print('loss_weight: ', loss_weight)
                print('loss_b: ', loss_b)
                print('loss_d: ', loss_d)
                raise NameError('loss_weight')
                
            loss_b_update = bias_criterion(logit_b, label)

            if np.isnan(loss_b_update.mean().item()):
                raise NameError('loss_b_update')

            loss_d_from_ori = criterion(logit_d, label)
            loss_d_from_adv = criterion(logit_d_adv, label)
            loss_d_update = loss_d_from_ori.to(device) * loss_weight.to(device) + loss_d_from_adv.to(device) * loss_weight_adv.to(device) #hacked version

            if np.isnan(loss_d_update.mean().item()):
                raise NameError('loss_d_update')
            loss = loss_b_update.mean() + loss_d_update.mean()
#             loss = loss_d_update.mean()

            # for a easier look up, we only check for client 1
            if client == 1:
                writer.add_image('adv images', data_adv[0], local_epochs*epochs+step)
                writer.add_image('ori images', data[0], local_epochs*epochs+step)


            bias_attr = attr[:, bias_attr_idx]

            aligned_mask = (label == bias_attr).cpu()
            skewed_mask = (label != bias_attr).cpu()
            # check biased model's performance
            
            # print('*** loss_aligned: ', loss_b[aligned_mask].mean())
            # print('*** loss_skewed: ', loss_b[skewed_mask].mean())
            
            # check if biased model is biased
            if aligned_mask.any().item():
                writer.add_scalar(f"loss_client_{client}/b_train_aligned", loss_b[aligned_mask].mean(), local_epochs*epochs+step)

            if skewed_mask.any().item():
                writer.add_scalar(f"loss_client_{client}/b_train_skewed", loss_b[skewed_mask].mean(), local_epochs*epochs+step)

            # writer.add_scalar("loss_client_1/b_train_val", loss_per_sample_b.mean(), local_epochs*epochs+step)
            writer.add_scalar(f"loss_client_{client}/b_train_val", loss_b.mean(), local_epochs*epochs+step)
            writer.add_scalar(f"loss_client_{client}/b_train_gce", loss_b_update.mean(), local_epochs*epochs+step)

            accuracy = torch.mean(evaluate(model_d, valid_loader_list[client]))
            writer.add_scalar(f"accuracy_client_{client}_local_debiased_model", accuracy, local_epochs*epoch+step)

                

            # if epochs > 120:
            #     model_b.eval()

            optimizer_b.zero_grad()
            optimizer_d.zero_grad()
            
            loss.backward()
            
            # if epochs < 120:
            #     optimizer_b.step() #line 10

            optimizer_b.step()
            optimizer_d.step()

#             batch_loss.append(loss.item())
            batch_loss.append(loss_d_update.nanmean().item())
            batch_loss_ori.append(loss_d_from_ori.nanmean().item())
            batch_loss_adv.append(loss_d_from_adv.nanmean().item())
        #------ finish updating a client's local model -------

        # epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
        # return model_b.state_dict(), model_d.state_dict(), sum(epoch_loss) / len(epoch_loss)

        return model_b.state_dict(), model_d.state_dict(), sum(batch_loss) / len(batch_loss), sum(batch_loss_ori) / len(batch_loss_ori), sum(batch_loss_adv) / len(batch_loss_adv)
    

    def evaluate(model, data_loader):
        model.eval()
        acc = 0
        attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
        for index, data, attr in tqdm(data_loader, leave=False):
            label = attr[:, target_attr_idx]
            data = data.to(device)
            attr = attr.to(device)
            label = label.to(device)
            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()

            attr = attr[:, [target_attr_idx, bias_attr_idx]]

            attrwise_acc_meter.add(correct.cpu(), attr.cpu())
            # attrwise_acc_meter.add(correct, attr)


        accs = attrwise_acc_meter.get_mean()

        model.train()

        return accs
    
    def disparate_impact_helper(digit, model, data_loader):
    #
    # disparate impact = ((num_correct_digit1(color=red))/  (num_digit1(color=red)))/((num_correct_digit1(color!=red))/  (num_digit1(color!=red)))
        model.eval()
        result = []
        for index, data, attr in tqdm(data_loader, leave=False):
            label = attr[:, target_attr_idx]
            data = data.to(device)
            attr = attr.to(device)
            label = label.to(device)
            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                unpriviledge_count = torch.sum((attr[:, 0] == digit) & (attr[:, 1] == digit)) # 1,1
                priviledge_count = torch.sum((attr[:, 0] == digit) & (attr[:, 1] != digit))   # 1,2
                
                unpriviledge_correct_count = torch.sum((pred == digit) & (attr[:, 0] == digit) & (attr[:, 1] == digit))
                priviledge_correct_count = torch.sum((pred == digit) & (attr[:, 0] == digit) & (attr[:, 1] != digit))

                disparate_impact = ((unpriviledge_correct_count / unpriviledge_count) / (priviledge_correct_count / priviledge_count)).item()

                if not (math.isnan(disparate_impact) or math.isinf(disparate_impact)):
                    result.append(disparate_impact)
                # else:
                #     print(f'### disparate impact error on digit {digit}')
                #     print('unpriviledge_count: ', unpriviledge_count)
                #     print('priviledge_count: ', priviledge_count)
                #     print('unpriviledge_correct_count', unpriviledge_correct_count)
                #     print('priviledge_correct_count: ', priviledge_correct_count)
                #     print()


        
        disparate_impact = np.mean(np.array(result))

        model.train()

        # print('disparate_impact on ' + str(digit) +': ', disparate_impact)

        return disparate_impact
    

        
            

    global_epochs = main_num_steps
    
    model_b_arr = {}
    for epoch in tqdm(range(global_epochs)):
        local_weights, local_losses, local_loss_ori, local_loss_adv = [], [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        model_global.train()
        # m = max(int(frac * num_users), 1)
        # idxs_users = np.random.choice(range(num_users), m, replace=False)
        idxs_users = [i for i in range(10)]

        for idx in idxs_users:
            try:
                model_b = model_b_arr[idx]
                # print('### old model_b is loaded###')
            except:
                model_b_arr[idx] = copy.deepcopy(model_biased)
                # model_b_arr[idx] = copy.deepcopy(model_biased)
                model_b = model_b_arr[idx]

            model_d = copy.deepcopy(model_global)

            w_b, w_d, loss, loss_from_ori, loss_from_adv = update_weights(model_b, model_d, idx, epoch)
            local_weights.append(copy.deepcopy(w_d))
            local_losses.append(copy.deepcopy(loss))
            local_loss_ori.append(copy.deepcopy(loss_from_ori))
            local_loss_adv.append(copy.deepcopy(loss_from_adv))

            # update local biased model weights
            model_b_arr[idx].load_state_dict(w_b)
            

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        model_global.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        loss_avg_from_ori = sum(local_loss_ori) / len(local_loss_ori)
        loss_avg_from_adv = sum(local_loss_adv) / len(local_loss_adv)



        # Calculate avg training accuracy over all users at every epoch
        
        # model_global.eval()
        # evaluate(model_global, valid_loader)
        # -- --
        main_log_freq = 1
        if epoch % main_log_freq == 0:
            print('acc: ', torch.mean(evaluate(model_global, valid_loader)))

            writer.add_scalar("loss_avg/two_term", loss_avg, epoch)
            writer.add_scalar("loss_avg/ori", loss_avg_from_ori, epoch)
            writer.add_scalar("loss_avg/adv", loss_avg_from_adv, epoch)
            writer.add_scalar("acc", torch.mean(evaluate(model_global, valid_loader)), epoch)
            
            print('loss: ', loss_avg)

            #disparate impact
            disparate_impact_arr = []
            for i in range(10):
                disparate_impact = disparate_impact_helper(i, model_global, valid_loader)
                
                if not np.isnan(disparate_impact):
                    disparate_impact_arr.append(disparate_impact)
                    

                    writer.add_scalar('disparate_impact/' + str(i), disparate_impact, epoch)   
                

            # print('mean_b: ', np.mean(np.array(disparate_impact_b_arr)))
            # print('mean_d: ', np.mean(np.array(disparate_impact_d_arr)))

            writer.add_scalar('disparate_impact_mean/', np.mean(np.array(disparate_impact_arr)), epoch)   
            
            
            # -- garbage ----

        # define evaluation function

    end_time = datetime.now()
    print(f'It takes {end_time - start_time} in the training')
        
            

