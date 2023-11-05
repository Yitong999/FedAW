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
from data.util import get_dataset, IdxDataset, ZippedDataset, average_weights, DatasetSplit, FedWt_v1, FedWt_v2
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
    
    train_target_attr = train_dataset.attr[:, target_attr_idx]
    train_bias_attr = train_dataset.attr[:, bias_attr_idx]
    attr_dims = []
    attr_dims.append(torch.max(train_target_attr).item() + 1)
    attr_dims.append(torch.max(train_bias_attr).item() + 1)
    num_classes = attr_dims[0]

    data_set_tag_list = ['ColoredMNIST-Skewed0.005-Severity1', 'ColoredMNIST-Skewed0.005-Severity2', 
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
            batch_size=256,
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
        
        score = 0 # score for weight
        for step in tqdm(range(local_epochs)):
            batch_loss = []

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
            

            

            
            
            logit_b = model_b(data)
            if np.isnan(logit_b.mean().item()):
                print(logit_b)
                raise NameError('logit_b')
            logit_d = model_d(data)
            
            loss_b = criterion(logit_b, label).cpu().detach()
            loss_d = criterion(logit_d, label).cpu().detach()
            
            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d')
            
            loss_per_sample_b = loss_b
            loss_per_sample_d = loss_d

            '''undo EMa
            # EMA sample loss
            sample_loss_ema_b.update(loss_b, index)
            sample_loss_ema_d.update(loss_d, index)

            # class-wise normalize
            loss_b = sample_loss_ema_b.parameter[index].clone().detach()
            loss_d = sample_loss_ema_d.parameter[index].clone().detach()

            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b_ema')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d_ema')
            
            label_cpu = label.cpu()
            
            for c in range(num_classes):
                class_index = np.where(label_cpu == c)[0]
                max_loss_b = sample_loss_ema_b.max_loss(c)
                max_loss_d = sample_loss_ema_d.max_loss(c)
                loss_b[class_index] /= max_loss_b
                loss_d[class_index] /= max_loss_d
            '''

            loss_weight = loss_b / (loss_b + loss_d + 1e-8)
            score += loss_weight.mean().item() # assign value as score metrics
            
            if step == 9:
                print('score: ', score)
            
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


            loss_d_update = criterion(logit_d, label) * loss_weight.to(device)

            if np.isnan(loss_d_update.mean().item()):
                raise NameError('loss_d_update')
            loss = loss_b_update.mean() + loss_d_update.mean()


            bias_attr = attr[:, bias_attr_idx]

            aligned_mask = (label == bias_attr).cpu()
            skewed_mask = (label != bias_attr).cpu()

            
            # check if biased model is biased
            if aligned_mask.any().item():
                writer.add_scalar(f"loss_client_{client}/b_train_aligned", loss_per_sample_b[aligned_mask].mean(), local_epochs*epochs+step)

            if skewed_mask.any().item():
                writer.add_scalar(f"loss_client_{client}/b_train_skewed", loss_per_sample_b[skewed_mask].mean(), local_epochs*epochs+step)

            # writer.add_scalar("loss_client_1/b_train_val", loss_per_sample_b.mean(), local_epochs*epochs+step)
            writer.add_scalar(f"loss_client_{client}/b_train_ce", loss_per_sample_b.mean(), local_epochs*epochs+step)
            writer.add_scalar(f"loss_client_{client}/b_train_gce", loss_b_update.mean(), local_epochs*epochs+step)

            accuracy = torch.mean(evaluate(model_d, valid_loader_list[client]))
            writer.add_scalar(f"accuracy_client_{client}_local_debiased_model", accuracy, local_epochs*epoch+step)


            optimizer_b.zero_grad()
            optimizer_d.zero_grad()
            
            loss.backward()
            
            # if epochs < 120:
            #     optimizer_b.step() #line 10

            optimizer_b.step()
            optimizer_d.step()

#             batch_loss.append(loss.item())
            batch_loss.append(loss_d_update.nanmean().item())
        #------ finish updating a client's local model -------

        # epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
        # return model_b.state_dict(), model_d.state_dict(), sum(epoch_loss) / len(epoch_loss)

        return model_b.state_dict(), model_d.state_dict(), sum(batch_loss) / len(batch_loss), score
    

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
        local_weights, local_losses, scores, local_loss_ori, local_loss_adv = [], [], [], [], []
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

            w_b, w_d, loss, score = update_weights(model_b, model_d, idx, epoch)
            local_weights.append(copy.deepcopy(w_d))
            local_losses.append(copy.deepcopy(loss))
            scores.append(score)
            # update local biased model weights
            model_b_arr[idx].load_state_dict(w_b)
            

        # update global weights
        # global_weights = average_weights(local_weights)
        global_weights = FedWt_v1(local_weights, scores)

        # update global weights
        model_global.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)



        # Calculate avg training accuracy over all users at every epoch

        # -- --
        main_log_freq = 1
        if epoch % main_log_freq == 0:
            print('acc: ', torch.mean(evaluate(model_global, valid_loader)))

            writer.add_scalar("loss_avg", loss_avg, epoch)

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
        
            

