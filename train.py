import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from resmasknet_test import *
import random
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np

lr = 0.01
weight_decay = 0.001
momentum = 0.9


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
        map_location={"cuda:0": "cpu"},
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
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Define the fully connected layers on top of concatenated feature vectors
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 7)
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
        self.emotion = 'happy'
        self.idx = self.FER_2013_EMONUM[self.emotion]
        
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.Tanh()
        # self.dropout = nn.Dropout(p=0.5)
        # self.relu = nn.ReLU()
        
    
    # _once
    def forward_once(self, x):
        # Forward pass through ResMaskNet
        x = self.model(x)
        x = x.view(x.size()[0], -1)
        # x = self.activation(x)
        return x
    
    def forward(self, x1, x2):
        # Pass each input image through ResMaskNet to obtain feature vectors
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        
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
    



aver_4_sorted_data = ['ha_212.png', 'ha_393.png', 'ha_428.png', 'ha_489.png', 'ha_412.png', 'ha_202.png', 'ha_348.png', 'ha_24.png', 'ha_407.png', 'ha_288.png', 'ha_367.png', 'ha_341.png', 'ha_235.png', 'ha_443.png', 'ha_450.png', 'ha_185.png', 'ha_50.png', 'ha_491.png', 'ha_301.png', 'ha_11.png', 'ha_422.png', 'ha_130.png', 'ha_243.png', 'ha_201.png', 'ha_32.png', 'ha_19.png', 'ha_384.png', 'ha_184.png', 'ha_311.png', 'ha_497.png', 'ha_256.png', 'ha_27.png', 'ha_107.png', 'ha_268.png', 'ha_329.png', 'ha_315.png', 'ha_2.png', 'ha_368.png', 'ha_241.png', 'ha_303.png', 'ha_221.png', 'ha_151.png', 'ha_342.png', 'ha_296.png', 'ha_152.png', 'ha_442.png', 'ha_186.png', 'ha_344.png', 'ha_215.png', 'ha_320.png', 'ha_149.png', 'ha_122.png', 'ha_54.png', 'ha_476.png', 'ha_106.png', 'ha_249.png', 'ha_132.png', 'ha_33.png', 'ha_207.png', 'ha_451.png', 'ha_172.png', 'ha_244.png', 'ha_454.png', 'ha_43.png', 'ha_131.png', 'ha_377.png', 'ha_396.png', 'ha_284.png', 'ha_59.png', 'ha_1.png', 'ha_252.png', 'ha_466.png', 'ha_110.png', 'ha_404.png', 'ha_292.png', 'ha_124.png', 'ha_482.png', 'ha_477.png', 'ha_5.png', 'ha_382.png', 'ha_9.png', 'ha_334.png', 'ha_381.png', 'ha_111.png', 'ha_380.png', 'ha_310.png', 'ha_475.png', 'ha_128.png', 'ha_314.png', 'ha_262.png', 'ha_174.png', 'ha_119.png', 'ha_139.png', 'ha_257.png', 'ha_233.png', 'ha_116.png', 'ha_399.png', 'ha_84.png', 'ha_145.png', 'ha_16.png']
base_folder = 'data/happiness_selected_imgonly100/'
data_path = [base_folder + i for i in aver_4_sorted_data]

def generate_dataset(data_path):
    dataset = []
    for i in range(len(data_path)):
        for j in range(i+1, len(data_path)):
            dataset.append([[data_path[i], data_path[j]], 1])
    return dataset

raw_data = generate_dataset(data_path)
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# dataset = PairwiseRatingDataset(raw_data, transform=transform)

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

BATCH_SIZE = 32
# Create DataLoader instances for train and val sets
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

loss_func = nn.BCELoss()
loss_func.to(device)

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

# start training
model.train()


def GetLoss(model, batch):
    batch = {k:v.to(model.device) for k, v in batch.items()}
    print(batch)
#     out = model(x1 = batch[])

optimizer = RAdam(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

import json

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    train_acc = 0.
    last_loss = 0.
    
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False):
        
        imgs1, imgs2, labels = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True), data[2].cuda(non_blocking=True)
        labels = labels.unsqueeze(1)
        labels = labels.float()
    
        optimizer.zero_grad()
        outputs = model(imgs1, imgs2)
        
        loss = loss_func(outputs, labels)
        acc = accuracy(outputs, labels)[0]
        acc = acc.sum() / len(acc)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        train_acc += acc.item()
        
        if i % int(len(train_dataloader) / 4) == int(len(train_dataloader) / 4) - 1:
            last_loss = running_loss / int(len(train_dataloader) / 4) # loss per 1/4 batch
            last_acc = train_acc / int(len(train_dataloader) / 4)
            print(' batch {} loss: {}, acc: {}'.format(i+1, last_loss, last_acc))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            tb_writer.add_scalar('Acc/train', last_acc, tb_x)
            running_loss = 0.
            train_acc = 0.
    return last_loss, last_acc


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
# Define the path and name of your log file
logfile = timestamp + "log.json"

# Define an empty dictionary object to store your log data
logdata = {}

epoch_number = 0

EPOCHS = 30

best_vloss = 1_000_000.
best_vacc = -1.

for epoch in range(EPOCHS):
    print('\nEPOCH {}:'.format(epoch_number + 1))
    
    # Make sure gradient tracking is on, and do a pass over the data
    model.train()
    avg_loss, avg_acc = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.eval()

    running_vloss = 0.0
    
    # eval
    vacc = 0.
    for i, vdata in tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False):
        with torch.no_grad():
            vimgs1, vimgs2, vlabels = vdata[0].cuda(non_blocking=True), vdata[1].cuda(non_blocking=True), vdata[2].cuda(non_blocking=True)
            vlabels = vlabels.unsqueeze(1)
            vlabels = vlabels.float()

            voutputs = model(vimgs1, vimgs2)
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
    
    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.add_scalars('Training vs. Validation Acc',
                    { 'Training' : avg_acc, 'Validation' : avg_vacc },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    # use acc instead
    if avg_vacc > best_vacc:
        best_vacc = avg_vacc
        model_path = 'model_{}_epoch{}.pt'.format(timestamp, epoch_number + 1)
        if not os.path.exists('check_points'):
            os.mkdir('check_points')
            
        model_path = os.path.join('check_points', model_path)
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
    
    epoch_number += 1

        
# Write the dictionary object to your log file as JSON
if not os.path.isfile(logfile):
    json.dump(logdata, logfile)
else:
    with open(logfile) as feedsjson:
        feeds = json.load(feedsjson)

    feeds.append(logdata)
    with open(logfile, mode='w') as f:
        f.write(json.dumps(feeds, indent=2))