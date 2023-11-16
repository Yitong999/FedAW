import os
import pickle
from tqdm import tqdm
from datetime import datetime
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, WeightedRandomSampler
from torch.utils.data.dataset import Subset
from torchvision import transforms as T
import torch.nn.functional as F

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

from config import ex
from data.util import get_dataset, IdxDataset, ZippedDataset,average_weights, DatasetSplit, FedWt_v1, FedWt_v2
from module.util import get_model
from module.loss import GeneralizedCELoss
from util import MultiDimAverageMeter

import math

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

    print(dataset_tag)

    device = torch.device(device)
    start_time = datetime.now()
    writer = SummaryWriter(os.path.join(log_dir, "summary", main_tag))

    train_dataset = get_dataset(
            dataset_tag,
            data_dir=data_dir,
            dataset_split="train",
            transform_split="train",
        )
    
   
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

    attr_dims = [10, 10]

    num_classes = attr_dims[0]

    
    valid_dataset = IdxDataset(valid_dataset)
    

    # define model and optimizer
    model_global = get_model(model_tag, attr_dims[0]).to(device)
    model_global_initial = copy.deepcopy(model_global)
    
    model = get_model(model_tag, num_classes).to(device)
    
    model_biased = get_model(model_tag, num_classes).to(device)

    
    # Training
    def update_weights(model_b, model, client, epoch, local_epochs=20):
        # Set mode to train model
        model.train()
        
        epoch_loss = []

        
        if main_optimizer_tag == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=main_learning_rate,
                weight_decay=main_weight_decay,
                momentum=0.9,
            )

            optimizer_b = torch.optim.SGD(
                model_b.parameters(),
                lr=main_learning_rate,
                weight_decay=main_weight_decay,
                momentum=0.9,
            )
            
        elif main_optimizer_tag == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=main_learning_rate,
                weight_decay=main_weight_decay,
            )

            optimizer_b = torch.optim.Adam(
                model_b.parameters(),
                lr=main_learning_rate,
                weight_decay=main_weight_decay,
            )
            
        elif main_optimizer_tag == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=main_learning_rate,
                weight_decay=main_weight_decay,
            )
            optimizer_b = torch.optim.AdamW(
                model_b.parameters(),
                lr=main_learning_rate,
                weight_decay=main_weight_decay,
            )
        else:
            raise NotImplementedError
        
        # define loss
        criterion = nn.CrossEntropyLoss(reduction='none')
        bias_criterion = GeneralizedCELoss()
        '''
        sample_loss_ema_b = EMA(torch.LongTensor(train_target_attr), alpha=0.7)
        sample_loss_ema_d = EMA(torch.LongTensor(train_target_attr), alpha=0.7)
        '''
        score = 0
        for step in tqdm(range(local_epochs)):
            batch_loss = []

            # train main model
            try:
                _, data, attr = next(train_iter)
            except:
                loader = train_loader_list[client]
                print('### client: ', client)
                train_iter = iter(loader)
                # train_iter = iter(train_loader)
                _, data, attr = next(train_iter)

            #original version
            data = data.to(device)
            attr = attr.to(device)
            label = attr[:, target_attr_idx]
            
            
            accuracy = torch.mean(evaluate(model, valid_loader_list[client]))
            writer.add_scalar(f"accuracy_of_local_model_on_{client}", accuracy, local_epochs*epoch+step)




            logit = model(data)

            logit_b = model_b(data)
            # loss_per_sample = criterion(logit.squeeze(1), label)

            loss_b = criterion(logit_b, label).cpu().detach()
            loss = criterion(logit, label).cpu().detach()

            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b')
            if np.isnan(loss.mean().item()):
                raise NameError('loss_d')
            
            loss_weight = loss_b / (loss_b + loss + 1e-8)
            score += loss_weight.mean().item() # assign value as score metrics



            loss_b_update = bias_criterion(logit_b, label)
            loss_d_update = criterion(logit, label)

            bias_attr = attr[:, bias_attr_idx]

            aligned_mask = (label == bias_attr).cpu()
            skewed_mask = (label != bias_attr).cpu()

            
            # check if biased model is biased
            if aligned_mask.any().item():
                writer.add_scalar(f"loss_client_{client}/b_train_aligned", loss_b[aligned_mask].mean(), local_epochs*epoch+step)

            if skewed_mask.any().item():
                writer.add_scalar(f"loss_client_{client}/b_train_skewed", loss_b[skewed_mask].mean(), local_epochs*epoch+step)

            # writer.add_scalar("loss_client_1/b_train_val", loss_per_sample_b.mean(), local_epochs*epochs+step)
            writer.add_scalar(f"loss_client_{client}/b_train_ce", loss_b.mean(), local_epochs*epoch+step)
            writer.add_scalar(f"loss_client_{client}/b_train_gce", loss_b_update.mean(), local_epochs*epoch+step)

            
            loss_sum = loss_b_update.mean() + loss_d_update.mean()


            optimizer.zero_grad()
            optimizer_b.zero_grad()


            loss_sum.backward()

            optimizer.step()
            optimizer_b.step()
    
            batch_loss.append(loss.mean().item())
        #------ finish updating a client's local model -------

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
        
        # if client == 1:
        #     disparate_impact_array = []
        #     for i in range(10):
        #         disparate_impact = disparate_impact_helper(i, model, valid_loader)


        #         disparate_impact_array.append(disparate_impact)


        #         writer.add_scalar('disparate_impact_client_1/' + str(i), disparate_impact, epoch)  
            
        
        return model_b.state_dict(), model.state_dict(), sum(epoch_loss) / len(epoch_loss), score
    
    
    # define evaluation function
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


        
        disparate_impact = np.mean(np.array(result))

        model.train()

        # print('disparate_impact on ' + str(digit) +': ', disparate_impact)

        return disparate_impact

    
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    

    global_epochs = main_num_steps
    num_users = 10
    frac = 1

    model_b_arr = {}
    idxs_users = [i for i in range(10)]
    for epoch in tqdm(range(global_epochs)):
        scores = []
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        model_global.train()
        m = max(int(frac * num_users), 1)
        # idxs_users = np.random.choice(range(num_users), m, replace=False)
        
        

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
            model_b_arr[idx].load_state_dict(w_b)
            scores.append(score)

       

        # update global weights
        if epoch <= 100:
            global_weights = average_weights(local_weights, scores)
        else:
            global_weights = FedWt_v1(local_weights, scores)

        # update global weights
        model_global.load_state_dict(global_weights)
        
        # if epoch == 2:
        #     # Save the entire model
        #     torch.save(model_global, 'models/model_global.pth')
        #     print('### saved global model')

        #     for idx in idxs_users:
        #         torch.save(model_b_arr[idx], f'models/model_b_{idx}.pth')
        #     print('### saved local biased model')





        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        
        # model_global.eval()
        # evaluate(model_global, valid_loader)
        # -- --
        main_log_freq = 5
        if epoch % main_log_freq == 0:
            print('acc: ', torch.mean(evaluate(model_global, valid_loader)))

            writer.add_scalar("loss_avg", loss_avg, epoch)
            writer.add_scalar("acc", torch.mean(evaluate(model_global, valid_loader)), epoch)
            
            print('loss: ', loss_avg)

            #disparate impact
            disparate_impact_arr = []
            for i in range(10):
                disparate_impact = disparate_impact_helper(i, model_global, valid_loader)
                

                disparate_impact_arr.append(disparate_impact)
                

                # writer.add_scalar('disparate_impact/' + str(i), disparate_impact, epoch)   
                

            # print('mean_b: ', np.mean(np.array(disparate_impact_b_arr)))
            # print('mean_d: ', np.mean(np.array(disparate_impact_d_arr)))

            writer.add_scalar('disparate_impact_mean/', np.mean(np.array(disparate_impact_arr)), epoch)   
            
            
            # -- garbage ----

        # define evaluation function


