import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
# from torch.nn.parallel import DistributedDataParallel
from resmasknet_test import *
import random
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
from tensorboardX import SummaryWriter
import json
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from human_data import feat_order, average_human_order, base_folders

lr = 5e-5
weight_decay = 1e-5
momentum = 0.9
# TOTAL epoch
EPOCHS = 25
val_start_epoch = 0
include_neg = True

BATCH_SIZE = 128

# prefix = 'cleaned_88_img'
# prefix = 'py_feat_100_happy_img'
CURR_EMO = 'disgust'
prefix = CURR_EMO.capitalize() + '_Rank1_50_include_neg'

print('lr: {}\nBATCH_SIZE: {}\nweight_decay: {}\nmomentum: {}\nEPOCHS: {}'.format(lr, BATCH_SIZE, weight_decay, momentum, EPOCHS))

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(1)


def resmasking_dropout1(in_channels=3, num_classes=7, weight_path=""):
    model = ResMasking(weight_path)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(512, 7)
        # nn.Linear(512, num_classes)
    )
    def get_resource_path():
        return ''

    # use_gpu = torch.cuda.is_available()
    # if use_gpu:
    #     model.load_state_dict(
    #         torch.load(
    #             os.path.join(
    #                 get_resource_path(), "ResMaskNet_Z_resmasking_dropout1_rot30.pth"
    #                 )
    #             )['net']
    #         )
    #     model.cuda()

    # else:
    model.load_state_dict(
        torch.load(
            os.path.join(
                get_resource_path(), "ResMaskNet_Z_resmasking_dropout1_rot30.pth"
            ),
        map_location={"cuda:0": "cuda"},
        )['net']
    )
    # model.fc = nn.Sequential(
    #     nn.Dropout(0.4),
    #     nn.Linear(512, 1)
    #     # nn.Linear(512, num_classes)
    # )
    return model

class SiameseRankNet(nn.Module):
    def __init__(self):
        super(SiameseRankNet, self).__init__()
        # Load ResMaskNet model
        self.model = resmasking_dropout1(in_channels=3, num_classes=7)
        # freeze
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # unfreeze
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Define the fully connected layers on top of concatenated feature vectors
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )
        self.FER_2013_EMO_DICT = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "sad",
            5: "surprise",
            6: "neutral",
        }
        self.FER_2013_EMONUM = {v:k for k, v in self.FER_2013_EMO_DICT.items()}
        self.emotion = CURR_EMO
        self.idx = self.FER_2013_EMONUM[self.emotion]
        
        self.sigmoid = nn.Sigmoid()
        # self.activation = nn.Tanh()
        # self.dropout = nn.Dropout(p=0.5)
        # self.relu = nn.ReLU()
        
    
    # _once
    def forward_once(self, x):
        # Forward pass through ResMaskNet
        x = self.model(x)
        x = x.view(x.size()[0], -1)
        # x = self.activation(x)
        return x
    
    def forward(self, x):
        # Pass each input image through ResMaskNet to obtain feature vectors
        x1 = self.forward_once(x[0])
        x2 = self.forward_once(x[1])
        
        # get target emotion idx
        x1 = x1[:, self.idx].unsqueeze(1)
        x2 = x2[:, self.idx].unsqueeze(1)

        # Concatenate the feature vectors
        # x = torch.cat((x1, x2), dim=1)

        # Pass the concatenated feature vector through the fully connected layers
        # x = self.fc(x)

        # Pass the output through sigmoid to obtain the probability of the input images being similar
        # normalize x1 - x2 as a probability that x1 should rank higher than x2
        x = self.sigmoid(x1 - x2)
        return x
    


# test for pyfeat ranking
aver_sorted_data = average_human_order[CURR_EMO]

# test for pyfeat ranking
cleaned_data = feat_order[CURR_EMO][:50]
aver_sorted_data = [i for i in aver_sorted_data if i in cleaned_data]


base_folder = base_folders[CURR_EMO]
print(base_folder)
data_path = [base_folder + i for i in aver_sorted_data]

def generate_dataset(data_path):
    dataset = []
    for i in range(len(data_path)):
        for j in range(i+1, len(data_path)):
            dataset.append([[data_path[i], data_path[j]], 1])
    return dataset

# all category data
g_data, b_data = feat_order[CURR_EMO][:50], feat_order[CURR_EMO][50:]

def generate_dataset_with_neg(good_set, bad_set, base_folder):
    dataset = []
    for i in range(len(good_set)):
        for j in range(i+1, len(good_set)):
            dataset.append([[good_set[i], good_set[j]], 1])

    for i in good_set:
        for j in bad_set:
            dataset.append([[i, j], 1])
    for i in range(len(dataset)):
        dataset[i][0] = [os.path.join(base_folder, dataset[i][0][j]) for j in range(len(dataset[i][0]))]
    return dataset


if not include_neg:
    # not include negative iamges
    raw_data = generate_dataset(data_path)
else:
    # include negative images
    print('including negative images')
    raw_data_for_valid = generate_dataset(data_path)
    raw_data = generate_dataset_with_neg(g_data, b_data, base_folder)

# print(raw_data)





class PairwiseRatingDataset(Dataset):
    def __init__(self, data, transform=None, mode='train'):
        
        self.data = data
        self.transform = transform

        # compute pairs and labels
        self.pairs = [i[0] for i in self.data]
        self.labels = [i[1] for i in self.data]

        # self.pairs = self.load_image_data()
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1 = Image.open(self.pairs[idx][0])
        img2 = Image.open(self.pairs[idx][1])
        
        # pre computed face box
        start_x, start_y, end_x, end_y = 193, 114, 442, 363
    
        img1 = img1.crop([start_x, start_y, end_x, end_y])
        img2 = img2.crop([start_x, start_y, end_x, end_y])

        # Apply transformations if any
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, self.labels[idx]


    # def __getitem__(self, idx):
    #     return self.pairs[idx][0], self.pairs[idx][1], self.labels[idx]

    # def load_image_data(self):
    #     print('loading image data...')
    #     # Load images and label for a given index
    #     image_pairs = []
        
    #     for i in range(self.__len__()):
    #         img1 = Image.open(self.pairs[i][0])
    #         img2 = Image.open(self.pairs[i][1])

    #         # Apply transformations if any
    #         if self.transform:
    #             img1 = self.transform(img1)
    #             img2 = self.transform(img2)
            
    #         image_pairs.append([img1, img2])

    #     return image_pairs
        

# Define transformations to be applied to images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# train_features, train_labels = next(iter(dataloader))



model = SiameseRankNet()

# speed up
# model = torch.nn.DataParallel(model, device_ids=[0, 1])


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    raise 'CUDA is not available'
model = model.cuda()
# .to(device)

# parallel training
model = nn.DataParallel(model)
# torch.distributed.init_process_group(backend="nccl")
# model = DistributedDataParallel(model) # device_ids will include all GPU devices by default


# dataset = PairwiseRatingDataset(raw_data, transform=transform)

def split_data(data):
    # Split data into train, val sets
    num_data = len(raw_data)
    num_train = int(0.8 * num_data)
    num_val = num_data - num_train

    # Create indices for train and val sets
    indices = list(range(num_data))
    random.shuffle(indices)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    # Create train and val datasets by indexing the PairwiseRatingDataset instance
    train_dataset = [raw_data[i] for i in train_indices]
    val_dataset = [raw_data[i] for i in val_indices]
    train_dataset = PairwiseRatingDataset(train_dataset, transform=transform)
    val_dataset = PairwiseRatingDataset(val_dataset, transform=transform)

    return train_dataset, val_dataset

bak_train_dataset, bak_val_dataset = split_data(raw_data_for_valid)
    

# Split data into train, val sets
num_data = len(raw_data)
num_train = int(0.8 * num_data)
num_val = num_data - num_train

# Create indices for train and val sets
indices = list(range(num_data))
random.shuffle(indices)
train_indices = indices[:num_train]
val_indices = indices[num_train:]

# Create train and val datasets by indexing the PairwiseRatingDataset instance
train_dataset = [raw_data[i] for i in train_indices]
val_dataset = [raw_data[i] for i in val_indices]
train_dataset = PairwiseRatingDataset(train_dataset, transform=transform)
val_dataset = PairwiseRatingDataset(val_dataset, transform=transform)

# Create DistributedSampler to handle distributing the dataset across nodes when training
# This can only be called after torch.distributed.init_process_group is called
# train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)


# Create DataLoader instances for train and val sets
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers = 8, pin_memory = True, drop_last=True, shuffle=True) # num_works = 8, pin_memory = True, drop_last=True,
# train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=(train_sampler is None), num_workers=8, pin_memory=False, sampler=train_sampler)

# val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# use good vs good only for val
val_dataloader = DataLoader(bak_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

loss_func = nn.BCELoss()
# loss_func = nn.CrossEntropyLoss()
loss_func.to(device)

# start training
model.train()

from radam import *

def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        # pred = torch.argmax(output, dim=1) # return the index of the max value in output
        # correct = pred.eq(target).float().sum(0)
        correct = (output > 0.5).sum(0)
        acc = correct * 100 / batch_size # acc percentage
        # print('acc', acc)
    return [acc]




def GetLoss(model, batch):
    batch = {k:v.to(model.device) for k, v in batch.items()}
    print(batch)
#     out = model(x1 = batch[])

optimizer = RAdam(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )



def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    train_acc = 0.
    last_loss = 0.
    loss_list = []
    acc_list = []
    
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False):
        
        imgs1, imgs2, labels = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True), data[2].cuda(non_blocking=True)

        labels = labels.unsqueeze(1)
        labels = labels.float()
    
        optimizer.zero_grad()
        outputs = model([imgs1, imgs2])
        
        loss = loss_func(outputs, labels)
        acc = accuracy(outputs, labels)[0]
        acc = acc.sum() / len(acc)

        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        train_acc += acc.item()
        
        loss_list.append(loss.item())
        acc_list.append(acc.item())
        
    avg_loss = sum(loss_list) / len(loss_list)
    avg_acc = sum(acc_list) / len(acc_list)

    return avg_loss, avg_acc


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('run/ranknet_trainer_{}_{}'.format(prefix,timestamp))
# Define the path and name of your log file
logfile = timestamp + "log.json"
logfile = os.path.join('check_points', prefix, timestamp, logfile)

# Define an empty dictionary object to store your log data
logdata = {}

best_vloss = 1_000_000.
best_vacc = -1.

# folders
if not os.path.exists('check_points'):
    os.mkdir('check_points')
if not os.path.exists(os.path.join('check_points', prefix, timestamp)):
    os.makedirs(os.path.join('check_points', prefix, timestamp))

# write config file in the dest folder
config = {
    CURR_EMO: CURR_EMO,
    lr: lr,
    weight_decay: weight_decay,
    momentum: momentum,
    # TOTAL epoch
    EPOCHS: EPOCHS,
    val_start_epoch: val_start_epoch,
    include_neg: include_neg,
    BATCH_SIZE:BATCH_SIZE,
    prefix: prefix,
    timestamp: timestamp,
}
conf_file = os.path.join('check_points', prefix, timestamp, "config.json")

if not os.path.isfile(conf_file):
    with open(conf_file, 'w') as f:
        json.dump(config, f)
        

for epoch in range(EPOCHS):
    epoch_number = epoch
    print('\nEPOCH {}:'.format(epoch_number + 1))
    
    # Make sure gradient tracking is on, and do a pass over the data
    model.train()
    avg_loss, avg_acc = train_one_epoch(epoch_number, writer)
    print(f'\nTraining LOSS: {avg_loss}')
    print(f'Training ACC: {avg_acc}')
    # We don't need gradients on to do reporting
    model.eval()

    running_vloss = 0.0
    
    # Log the running loss averaged per batch
    writer.add_scalar('EPOCH ACC/train', avg_acc, epoch_number + 1)
    writer.add_scalar('EPOCH Loss/train', avg_loss, epoch_number + 1)
    # eval
    # DEBUG mode
    if epoch + 1 > val_start_epoch or True:
        vacc = 0.
        for i, vdata in tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False):
            with torch.no_grad():
                vimgs1, vimgs2, vlabels = vdata[0].cuda(non_blocking=True), vdata[1].cuda(non_blocking=True), vdata[2].cuda(non_blocking=True)
                vlabels = vlabels.unsqueeze(1)
                vlabels = vlabels.float()

                voutputs = model([vimgs1, vimgs2])
                vloss = loss_func(voutputs, vlabels)
                running_vloss += vloss
                acc = accuracy(voutputs, vlabels)[0]
                acc = acc.sum() / len(acc)
                vacc += acc
            
        avg_vacc = vacc / (i + 1)
        avg_vloss = running_vloss / (i + 1)
        print('Result of EPOCH', epoch_number + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('ACC train {} valid {}'.format(avg_acc, avg_vacc))
            
        writer.add_scalar('EPOCH ACC/valid', avg_vacc, epoch_number + 1)
        writer.add_scalar('EPOCH Loss/valid', avg_vloss, epoch_number + 1)
    
    
        writer.add_scalars('Training vs. Validation Loss/EPOCH',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.add_scalars('Training vs. Validation Acc/EPOCH',
                        { 'Training' : avg_acc, 'Validation' : avg_vacc },
                        epoch_number + 1)
    

        # Track best performance, and save the model's state
        # use acc instead
        if avg_vacc > best_vacc or avg_vacc > 0.7:
            best_vacc = avg_vacc
            model_path = 'model_{}_epoch{}.pt'.format(timestamp, epoch_number + 1)
            model_path = os.path.join('check_points', prefix, timestamp, model_path)
            torch.save(model.state_dict(), model_path)
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss
        #     model_path = 'model_{}_epoch{}'.format(timestamp, epoch_number + 1)
        #     torch.save(model.state_dict(), model_path)

        # Store the values in a sub-dictionary with epoch number as key
        logdata[epoch_number + 1] = {
            "train_loss": avg_loss,
            "train_acc": avg_acc,
            "val_loss": avg_vloss.tolist(),
            "val_acc": avg_vacc.tolist()
        }
    writer.flush()
    epoch_number += 1


feeds = []
# Write the dictionary object to your log file as JSON
if not os.path.isfile(logfile):
    with open(logfile, 'w') as f:
        json.dump(logdata, f)
else:
    with open(logfile) as feedsjson:
        feeds = json.load(feedsjson)
    for k,v in logdata.items():
        feeds[k] = v
    with open(logfile, mode='w') as f:
        f.write(json.dumps(feeds, indent=2))

