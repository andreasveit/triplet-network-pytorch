from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import csv


class MNIST_t(data.Dataset):
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    train_triplet_file = 'train_triplets.txt'
    test_triplet_file = 'test_triplets.txt'

    def __init__(self, root,  n_train_triplets=50000, n_test_triplets=10000, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        
        self.transform = transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(root, self.processed_folder, self.training_file))
            self.make_triplet_list(n_train_triplets)
            triplets = []
            for line in open(os.path.join(root, self.processed_folder, self.train_triplet_file)):
                triplets.append((int(line.split()[0]), int(line.split()[1]), int(line.split()[2]))) # anchor, close, far
            self.triplets_train = triplets
        else:
            self.test_data, self.test_labels = torch.load(os.path.join(root, self.processed_folder, self.test_file))
            self.make_triplet_list(n_test_triplets)
            triplets = []
            for line in open(os.path.join(root, self.processed_folder, self.test_triplet_file)):
                triplets.append((int(line.split()[0]), int(line.split()[1]), int(line.split()[2]))) # anchor, close, far
            self.triplets_test = triplets


    def __getitem__(self, index):
        if self.train:
            idx1, idx2, idx3 = self.triplets_train[index]
            img1, img2, img3 = self.train_data[idx1], self.train_data[idx2], self.train_data[idx3]
        else:
            idx1, idx2, idx3 = self.triplets_test[index]
            img1, img2, img3 = self.test_data[idx1], self.test_data[idx2], self.test_data[idx3]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        if self.train:
            return len(self.triplets_train)
        else:
            return len(self.triplets_test)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def _check_triplets_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.train_triplet_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_triplet_file))

    def download(self):
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def make_triplet_list(self, ntriplets):

        if self._check_triplets_exists():
            return
        print('Processing Triplet Generation ...')
        if self.train:
            np_labels = self.train_labels.numpy()
            filename = self.train_triplet_file
        else:
            np_labels = self.test_labels.numpy()
            filename = self.test_triplet_file
        triplets = []
        for class_idx in range(10):
            a = np.random.choice(np.where(np_labels==class_idx)[0], int(ntriplets/10), replace=True)
            b = np.random.choice(np.where(np_labels==class_idx)[0], int(ntriplets/10), replace=True)
            while np.any((a-b)==0):
                np.random.shuffle(b)
            c = np.random.choice(np.where(np_labels!=class_idx)[0], int(ntriplets/10), replace=True)

            for i in range(a.shape[0]):
                triplets.append([int(a[i]), int(c[i]), int(b[i])])           

        with open(os.path.join(self.root, self.processed_folder, filename), "w") as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerows(triplets)
        print('Done!')





def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        return torch.LongTensor(labels)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        assert len(images) == length
        return torch.ByteTensor(images).view(-1, 28, 28)
