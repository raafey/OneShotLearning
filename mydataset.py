import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from torchvision import transforms
from PIL import Image

class Omniglot(Dataset):
    def __init__(self, data_path):
        super(Omniglot, self).__init__()
        np.random.seed(0)

        self.transform = transforms.Compose([transforms.RandomAffine(15), transforms.ToTensor()])
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        data = {}

        rot = [0, 90, 180, 270]

        for r in rot:
            for alphabet in os.listdir(data_path):
                data[alphabet] = {}
                alpha_path = os.path.join(data_path, alphabet)

                for character in os.listdir(alpha_path):
                    imgs = []
                    img_path = os.path.join(data_path, alphabet, character)
                    for img in os.listdir(img_path):
                        img_path = os.path.join(data_path, alphabet, character, img)
                        imgs.append(Image.open(img_path).rotate(r).convert('L'))
                    
                    data[alphabet][character] = imgs
        
        return data

    def __len__(self):
        return 21000000
    
    def get_len(self):
        len = 0
        for alphabet in self.data:
            for character in self.data[alphabet]:
                for img in character:
                    len += 1
        return len
    
    def get_num_classes(self):
        count = 0
        for alphabet in self.data:
            for character in self.data[alphabet]:
                count += 1
        return count

    def __getitem__(self, index):
        target = None
        img1 = None
        img2 = None

        alphabets = list(self.data.keys())


        if index % 2 == 0:
            target = torch.from_numpy(np.array([0.0], dtype=np.float32))

            alphabet_1 = random.choice(alphabets)
            alphabet_2 = random.choice(alphabets)

            while alphabet_1 == alphabet_2:
                alphabet_2 = random.choice(alphabets)
            
            characters_1 = list(self.data[alphabet_1].keys())
            characters_2 = list(self.data[alphabet_2].keys())
            char1 = random.choice(characters_1)
            char2 = random.choice(characters_2)

            print ((alphabet_1, char1))
            print ((alphabet_2, char2))

            img1 = random.choice(self.data[alphabet_1][char1])
            img2 = random.choice(self.data[alphabet_2][char2])
        else:
            target = torch.from_numpy(np.array([1.0], dtype=np.float32))
            alphabet = random.choice(alphabets)
            characters = list(self.data[alphabet].keys())
            character = random.choice(characters)
            imgs = self.data[alphabet][character]

            print ((alphabet, character))

            img1 = random.choice(imgs)
            img2 = random.choice(imgs)

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target


class OmniglotTrain(Dataset):
    def __init__(self, dataPath, transform=None):
        super(OmniglotTrain, self).__init__()
        np.random.seed(0)
        # self.dataset = dataset
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        agrees = [0, 90, 180, 270]
        idx = 0
        for agree in agrees:
            for alphaPath in os.listdir(dataPath):
                for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                    datas[idx] = []
                    for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                        filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                        datas[idx].append(Image.open(filePath).rotate(agree).convert('L'))
                    idx += 1
        print("finish loading training dataset to memory")
        return datas, idx

    def __len__(self):
        return  21000000

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


class OmniglotTest(Dataset):

    def __init__(self, dataPath, transform=None, times=200, way=20):
        np.random.seed(1)
        super(OmniglotTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        datas = {}
        idx = 0
        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    datas[idx].append(Image.open(filePath).convert('L'))
                idx += 1
        print("finish loading test dataset to memory")
        return datas, idx

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = random.choice(self.datas[self.c1])
            img2 = random.choice(self.datas[self.c1])
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = random.choice(self.datas[c2])

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        return img1, img2


# test
if __name__=='__main__':

   #omniglot = OmniglotTrain('./images_background', transforms.Compose([transforms.RandomAffine(15), transforms.ToTensor()]))
   omniglot = OmniglotTrain('./images_background')
   print (omniglot.__getitem__(21).shape)

   