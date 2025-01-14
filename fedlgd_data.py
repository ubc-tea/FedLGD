import torch
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import random
import os

from cifar10c_dataset import CIFAR10C, CIFAR10C_preprocessed
from data_utils import Cifar100CDataset



def prepare_data(args, im_size):

    if args.dataset == 'mnist':

        MEANS = [[0.1307, 0.1307, 0.1307]]
        STDS = [[0.3015, 0.3015, 0.3015]]

        # Prepare data
        transform_mnist = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[0], STDS[0])
            ])
        unnormalized_transform_mnist = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])


        # MNIST
        # mnist_trainset     = data_utils.DigitsDataset(data_path="./digit_data/digit_dataset/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist)
        # mnist_testset      = data_utils.DigitsDataset(data_path="./digit_data/digit_dataset/MNIST", channels=1, percent=args.percent, train=False, transform=transform_mnist)
        unnormalized_mnist_trainset = torchvision.datasets.MNIST(root="./digit_data", train=True, transform=unnormalized_transform_mnist, download=True)
        mnist_trainset = torchvision.datasets.MNIST(root="./digit_data", train=True, transform=transform_mnist, download=True)
        mnist_testset = torchvision.datasets.MNIST(root="./digit_data", train=False, transform=transform_mnist, download=True)
        # print(f'MNIST: {len(mnist_testset)}')


        mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
        mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False)
        

        train_loaders = [mnist_train_loader]
        test_loaders  = [mnist_test_loader]
        train_datasets = [mnist_trainset]
        test_datasets = [mnist_testset]

        min_data_len = 0

    elif args.dataset == 'svhn':

        MEANS = [[0.4379, 0.4440, 0.4731]]
        STDS = [[0.1161, 0.1192, 0.1017]]

        # Prepare data
        

        transform_svhn = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[0], STDS[0])
            ])
        unnormalized_transform_svhn = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor()
            ])

        

        # SVHN
        # svhn_trainset      = data_utils.DigitsDataset(data_path='./digit_data/digit_dataset/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn)
        # svhn_testset       = data_utils.DigitsDataset(data_path='./digit_data/digit_dataset/SVHN', channels=3, percent=args.percent,  train=False, transform=transform_svhn)
        unnormalized_svhn_trainset = torchvision.datasets.SVHN(root="./digit_data", split='train', transform=unnormalized_transform_svhn, download=True)
        svhn_trainset = torchvision.datasets.SVHN(root="./digit_data", split='train', transform=transform_svhn, download=True)
        svhn_testset = torchvision.datasets.SVHN(root="./digit_data", split='test', transform=transform_svhn, download=True)
        # print(f'SVHN: {len(svhn_testset)}')


        
        svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True)
        svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False)
        

        train_loaders = [svhn_train_loader]
        test_loaders  = [svhn_test_loader]
        train_datasets = [svhn_trainset]
        test_datasets = [svhn_testset]

        min_data_len = 0
    
    elif args.dataset == 'usps':

        MEANS = [[0.2473, 0.2473, 0.2473]]
        STDS = [[0.2665, 0.2665, 0.2665]]

        # Prepare data

        transform_usps = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[2], STDS[2])
            ])
        unnormalized_transform_usps = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])

        

        # USPS
        # usps_trainset      = data_utils.DigitsDataset(data_path='./digit_data/digit_dataset/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps)
        # usps_testset       = data_utils.DigitsDataset(data_path='./digit_data/digit_dataset/USPS', channels=1, percent=args.percent,  train=False, transform=transform_usps)
        unnormalized_usps_trainset = torchvision.datasets.USPS(root="./digit_data", train=True, transform=unnormalized_transform_usps, download=True)
        usps_trainset = torchvision.datasets.USPS(root="./digit_data", train=True, transform=transform_usps, download=True)
        usps_testset = torchvision.datasets.USPS(root="./digit_data", train=False, transform=transform_usps, download=True)
        # print(f'USPS: {len(usps_testset)}')

        
        usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True)
        usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False)
        

        train_loaders = [usps_train_loader]
        test_loaders  = [usps_test_loader]
        train_datasets = [usps_trainset]
        test_datasets = [usps_testset]

        min_data_len = 0

    elif args.dataset == 'digits':

        # MEANS = [[0.1307, 0.1307, 0.1307], [0.4379, 0.4440, 0.4731], [0.2473, 0.2473, 0.2473], [0.4828, 0.4603, 0.4320], [0.4595, 0.4629, 0.4097]]
        # STDS = [[0.3015, 0.3015, 0.3015], [0.1161, 0.1192, 0.1017], [0.2665, 0.2665, 0.2665], [0.1960, 0.1938, 0.1977], [0.1727, 0.1603, 0.1785]]

        MEANS = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        STDS = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]

        # MEANS = [[0.5, 0.5, 0.5], [0.4379, 0.4440, 0.4731], [0.5, 0.5, 0.5], [0.4828, 0.4603, 0.4320], [0.4595, 0.4629, 0.4097]]
        # STDS = [[0.5, 0.5, 0.5], [0.1161, 0.1192, 0.1017], [0.5, 0.5, 0.5], [0.1960, 0.1938, 0.1977], [0.1727, 0.1603, 0.1785]]

        # Prepare data
        transform_mnist = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[0], STDS[0])
            ])
        unnormalized_transform_mnist = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])

        transform_svhn = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[1], STDS[1])
            ])
        unnormalized_transform_svhn = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor()
            ])

        transform_usps = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[2], STDS[2])
            ])
        unnormalized_transform_usps = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])

        transform_synth = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[3], STDS[3])
            ])
        unnormalized_transform_synth = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor()
            ])

        transform_mnistm = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[4], STDS[4])
            ])
        unnormalized_transform_mnistm = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor()
            ])

        # MNIST
        # mnist_trainset     = data_utils.DigitsDataset(data_path="./digit_data/digit_dataset/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist)
        # mnist_testset      = data_utils.DigitsDataset(data_path="./digit_data/digit_dataset/MNIST", channels=1, percent=args.percent, train=False, transform=transform_mnist)
        unnormalized_mnist_trainset = torchvision.datasets.MNIST(root="./digit_data", train=True, transform=unnormalized_transform_mnist, download=True)
        mnist_trainset = torchvision.datasets.MNIST(root="./digit_data", train=True, transform=transform_mnist, download=True)
        mnist_testset = torchvision.datasets.MNIST(root="./digit_data", train=False, transform=transform_mnist, download=True)
        # print(f'MNIST: {len(mnist_testset)}')

        # SVHN
        # svhn_trainset      = data_utils.DigitsDataset(data_path='./digit_data/digit_dataset/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn)
        # svhn_testset       = data_utils.DigitsDataset(data_path='./digit_data/digit_dataset/SVHN', channels=3, percent=args.percent,  train=False, transform=transform_svhn)
        unnormalized_svhn_trainset = torchvision.datasets.SVHN(root="./digit_data", split='train', transform=unnormalized_transform_svhn, download=True)
        svhn_trainset = torchvision.datasets.SVHN(root="./digit_data", split='train', transform=transform_svhn, download=True)
        svhn_testset = torchvision.datasets.SVHN(root="./digit_data", split='test', transform=transform_svhn, download=True)
        # print(f'SVHN: {len(svhn_testset)}')

        # USPS
        # usps_trainset      = data_utils.DigitsDataset(data_path='./digit_data/digit_dataset/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps)
        # usps_testset       = data_utils.DigitsDataset(data_path='./digit_data/digit_dataset/USPS', channels=1, percent=args.percent,  train=False, transform=transform_usps)
        unnormalized_usps_trainset = torchvision.datasets.USPS(root="./digit_data", train=True, transform=unnormalized_transform_usps, download=True)
        usps_trainset = torchvision.datasets.USPS(root="./digit_data", train=True, transform=transform_usps, download=True)
        usps_testset = torchvision.datasets.USPS(root="./digit_data", train=False, transform=transform_usps, download=True)
        # print(f'USPS: {len(usps_testset)}')

        # Synth Digits
        # synth_trainset     = data_utils.DigitsDataset(data_path='./digit_data/digit_dataset/SynthDigits/', channels=3, percent=args.percent,  train=True,  transform=transform_synth)
        # synth_testset      = data_utils.DigitsDataset(data_path='./digit_data/digit_dataset/SynthDigits/', channels=3, percent=args.percent,  train=False, transform=transform_synth)
        unnormalized_synth_trainset     = ImageFolder('./digit_data/synthetic_digits/imgs_train', transform=unnormalized_transform_synth)
        synth_trainset     = ImageFolder('./digit_data/synthetic_digits/imgs_train', transform=transform_synth)
        synth_testset     = ImageFolder('./digit_data/synthetic_digits/imgs_valid', transform=transform_synth)
        # print(f'SYNTH: {len(synth_testset)}')

        # MNIST-M
        # mnistm_trainset     = data_utils.DigitsDataset(data_path='./digit_data/digit_dataset/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=transform_mnistm)
        # mnistm_testset      = data_utils.DigitsDataset(data_path='./digit_data/digit_dataset/MNIST_M/', channels=3, percent=args.percent,  train=False, transform=transform_mnistm)
        unnormalized_mnistm_trainset     = ImageFolder('./digit_data/mnistm/train', transform=unnormalized_transform_mnistm)
        mnistm_trainset     = ImageFolder('./digit_data/mnistm/train', transform=transform_mnistm)
        mnistm_testset     = ImageFolder('./digit_data/mnistm/test', transform=transform_mnistm)
        # print(f'MNISTM: {len(mnistm_testset)}')

        mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
        mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False)
        svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True)
        svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False)
        usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True)
        usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False)
        synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True)
        synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False)
        mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True)
        mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False)

        train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
        test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]
        unnormalized_train_datasets = [unnormalized_mnist_trainset, unnormalized_svhn_trainset, unnormalized_usps_trainset, unnormalized_synth_trainset, unnormalized_mnistm_trainset]
        train_datasets = [mnist_trainset, svhn_trainset, usps_trainset, synth_trainset, mnistm_trainset]
        test_datasets = [mnist_testset, svhn_testset, usps_testset, synth_testset, mnistm_testset]

        min_data_len = min(len(mnist_testset), len(svhn_testset), len(usps_testset), len(synth_testset), len(mnistm_testset))

    elif args.dataset == 'retina':

        MEANS = [[0.5594, 0.2722, 0.0819], [0.7238, 0.3767, 0.1002], [0.5886, 0.2652, 0.1481], [0.7085, 0.4822, 0.3445]]
        STDS = [[0.1378, 0.0958, 0.0343], [0.1001, 0.1057, 0.0503], [0.1147, 0.0937, 0.0461], [0.1663, 0.1541, 0.1066]]

        # data_base_path = './data/segmented_retina'
        data_base_path = './data/retina_balanced'
        
        transform_unnormalized = transforms.Compose([
                transforms.Resize(im_size),            
                transforms.ToTensor()
        ])
        
        # Drishti
        transform_drishti = transforms.Compose([
                transforms.Resize(im_size),            
                transforms.ToTensor(),
                transforms.Normalize(MEANS[0], STDS[0])
        ])
        drishti_train_path = os.path.join(data_base_path, 'Drishti', 'Training')
        drishti_test_path = os.path.join(data_base_path, 'Drishti', 'Testing')
        unnormalized_drishti_trainset = ImageFolder(drishti_train_path, transform=transform_unnormalized)
        drishti_trainset = ImageFolder(drishti_train_path, transform=transform_drishti)
        drishti_testset = ImageFolder(drishti_test_path, transform=transform_drishti)
        
        # kaggle
        transform_kaggle = transforms.Compose([
                transforms.Resize(im_size),            
                transforms.ToTensor(),
                transforms.Normalize(MEANS[1], STDS[1])
        ])
        kaggle_train_path = os.path.join(data_base_path, 'kaggle_arima', 'Training')
        kaggle_test_path = os.path.join(data_base_path, 'kaggle_arima', 'Testing')
        unnormalized_kaggle_trainset = ImageFolder(kaggle_train_path, transform=transform_unnormalized)
        kaggle_trainset = ImageFolder(kaggle_train_path, transform=transform_kaggle)
        kaggle_testset = ImageFolder(kaggle_test_path, transform=transform_kaggle)
        
        # RIM
        transform_rim = transforms.Compose([
                transforms.Resize(im_size),            
                transforms.ToTensor(),
                transforms.Normalize(MEANS[2], STDS[2])
        ])
        rim_train_path = os.path.join(data_base_path, 'RIM', 'Training')
        rim_test_path = os.path.join(data_base_path, 'RIM', 'Testing')
        unnormalized_rim_trainset = ImageFolder(rim_train_path, transform=transform_unnormalized)
        rim_trainset = ImageFolder(rim_train_path, transform=transform_rim)
        rim_testset = ImageFolder(rim_test_path, transform=transform_rim)
        
        # refuge
        transform_refuge = transforms.Compose([
                transforms.Resize(im_size),            
                transforms.ToTensor(),
                transforms.Normalize(MEANS[3], STDS[3])
        ])
        refuge_train_path = os.path.join(data_base_path, 'REFUGE', 'Training')
        refuge_test_path = os.path.join(data_base_path, 'REFUGE', 'Testing')
        unnormalized_refuge_trainset = ImageFolder(refuge_train_path, transform=transform_unnormalized)
        refuge_trainset = ImageFolder(refuge_train_path, transform=transform_refuge)
        refuge_testset = ImageFolder(refuge_test_path, transform=transform_refuge)
        

        #####################################
        Drishti_train_loader = torch.utils.data.DataLoader(drishti_trainset, batch_size=args.batch, shuffle=True)
        Drishti_test_loader = torch.utils.data.DataLoader(drishti_testset, batch_size=args.batch, shuffle=False)

        kaggle_train_loader = torch.utils.data.DataLoader(kaggle_trainset, batch_size=args.batch, shuffle=True)
        kaggle_test_loader = torch.utils.data.DataLoader(kaggle_testset, batch_size=args.batch, shuffle=False)

        rim_train_loader = torch.utils.data.DataLoader(rim_trainset, batch_size=args.batch, shuffle=True)
        rim_test_loader = torch.utils.data.DataLoader(rim_testset, batch_size=args.batch, shuffle=False)

        refuge_train_loader = torch.utils.data.DataLoader(refuge_trainset, batch_size=args.batch, shuffle=True)
        refuge_test_loader = torch.utils.data.DataLoader(refuge_testset, batch_size=args.batch, shuffle=False)
        
        train_loaders = [Drishti_train_loader, kaggle_train_loader, rim_train_loader, refuge_train_loader]
        test_loaders = [Drishti_test_loader, kaggle_test_loader, rim_test_loader, refuge_test_loader]
        unnormalized_train_datasets = [unnormalized_drishti_trainset, unnormalized_kaggle_trainset, unnormalized_rim_trainset, unnormalized_refuge_trainset]
        train_datasets = [drishti_trainset, kaggle_trainset, rim_trainset, refuge_trainset]
        test_datasets = [drishti_testset, kaggle_testset, rim_testset, refuge_testset]

        min_data_len = min(len(drishti_testset), len(kaggle_testset), len(rim_testset), len(refuge_testset))

    elif args.dataset == 'cifar10c':

        MEANS = [[0, 0, 0] for _ in range(57)]
        STDS = [[1, 1, 1] for _ in range(57)]

        transform_unnormalized = transforms.Compose([
                transforms.Resize(im_size),            
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((-30,30)),
                transforms.ToTensor()
        ])
        
        train_datasets = []
        test_datasets = []
        train_loaders = []
        test_loaders = []
        for i in range(57):
            trainset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed', train=True, client_num = i, transform=transform_unnormalized)
            testset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed', train=False, client_num = i, transform=transform_unnormalized)
            train_loader_tmp = torch.utils.data.DataLoader(trainset_tmp, batch_size=args.batch, shuffle=True)
            test_loader_tmp = torch.utils.data.DataLoader(testset_tmp, batch_size=args.batch, shuffle=False)
            train_datasets.append(trainset_tmp)
            test_datasets.append(testset_tmp)
            train_loaders.append(train_loader_tmp)
            test_loaders.append(test_loader_tmp)

        min_data_len = min([len(test_dataset) for test_dataset in test_datasets])

    elif args.dataset == 'cifar10c_alpha1':

        MEANS = [[0, 0, 0] for _ in range(57)]
        STDS = [[1, 1, 1] for _ in range(57)]

        transform_unnormalized = transforms.Compose([
                transforms.Resize(im_size),            
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((-30,30)),
                transforms.ToTensor()
        ])
        
        train_datasets = []
        test_datasets = []
        train_loaders = []
        test_loaders = []
        for i in range(57):
            trainset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed_alpha1', train=True, client_num = i, transform=transform_unnormalized)
            testset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed_alpha1', train=False, client_num = i, transform=transform_unnormalized)
            train_loader_tmp = torch.utils.data.DataLoader(trainset_tmp, batch_size=args.batch, shuffle=True)
            test_loader_tmp = torch.utils.data.DataLoader(testset_tmp, batch_size=args.batch, shuffle=False)
            train_datasets.append(trainset_tmp)
            test_datasets.append(testset_tmp)
            train_loaders.append(train_loader_tmp)
            test_loaders.append(test_loader_tmp)

        min_data_len = min([len(test_dataset) for test_dataset in test_datasets])
    
    elif args.dataset == 'cifar10c_alpha5':

        MEANS = [[0, 0, 0] for _ in range(57)]
        STDS = [[1, 1, 1] for _ in range(57)]

        transform_unnormalized = transforms.Compose([
                transforms.Resize(im_size),            
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((-30,30)),
                transforms.ToTensor()
        ])
        
        train_datasets = []
        test_datasets = []
        train_loaders = []
        test_loaders = []
        for i in range(57):
            trainset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed_alpha5', train=True, client_num = i, transform=transform_unnormalized)
            testset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed_alpha5', train=False, client_num = i, transform=transform_unnormalized)
            train_loader_tmp = torch.utils.data.DataLoader(trainset_tmp, batch_size=args.batch, shuffle=True)
            test_loader_tmp = torch.utils.data.DataLoader(testset_tmp, batch_size=args.batch, shuffle=False)
            train_datasets.append(trainset_tmp)
            test_datasets.append(testset_tmp)
            train_loaders.append(train_loader_tmp)
            test_loaders.append(test_loader_tmp)

        min_data_len = min([len(test_dataset) for test_dataset in test_datasets])
    
    elif args.dataset == 'cifar100c':

        client_num = 10

        MEANS = [[0, 0, 0] for _ in range(client_num)]
        STDS = [[1, 1, 1] for _ in range(client_num)]

        transform_unnormalized = transforms.Compose([
                transforms.Resize(im_size),            
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((-30,30)),
                transforms.ToTensor()
        ])
        
        train_datasets = []
        test_datasets = []
        train_loaders = []
        test_loaders = []
        for i in range(client_num):
            trainset_tmp = Cifar100CDataset(data_path=f'./data/numpy_dataloader/CIFAR100-C/cifar100c{client_num}c/', channels=3, partition=i, train=True, transform=transform_unnormalized)
            testset_tmp = Cifar100CDataset(data_path=f'./data/numpy_dataloader/CIFAR100-C/cifar100c{client_num}c/', channels=3, partition=i, train=False, transform=transform_unnormalized)
            train_loader_tmp = torch.utils.data.DataLoader(trainset_tmp, batch_size=args.batch, shuffle=True)
            test_loader_tmp = torch.utils.data.DataLoader(testset_tmp, batch_size=args.batch, shuffle=False)
            train_datasets.append(trainset_tmp)
            test_datasets.append(testset_tmp)
            train_loaders.append(train_loader_tmp)
            test_loaders.append(test_loader_tmp)

        min_data_len = min([len(test_dataset) for test_dataset in test_datasets])

    else:
        NotImplementedError


    # shuffled_idxes = [list(range(0, len(train_datasets[idx]))) for idx in range(len(train_datasets))]
    # for idx in range(len(shuffled_idxes)):
    #     random.shuffle(shuffled_idxes[idx])
    # cropped_train_sets = [torch.utils.data.Subset(train_datasets[idx], shuffled_idxes[idx][:min_data_len]) for idx in range(len(train_datasets))]
    # # concated_test_set = torch.utils.data.ConcatDataset(concated_test_set)
    # # print(len(concated_test_set))
    # cropped_train_loaders = [torch.utils.data.DataLoader(cropped_train_set, batch_size=args.batch, shuffle=False) for cropped_train_set in cropped_train_sets]

    # shuffled_idxes = [list(range(0, len(test_datasets[idx]))) for idx in range(len(test_datasets))]
    # for idx in range(len(shuffled_idxes)):
    #     random.shuffle(shuffled_idxes[idx])
    # cropped_test_sets = [torch.utils.data.Subset(test_datasets[idx], shuffled_idxes[idx][:min_data_len]) for idx in range(len(test_datasets))]
    # # concated_test_set = torch.utils.data.ConcatDataset(concated_test_set)
    # # print(len(concated_test_set))
    # cropped_test_loaders = [torch.utils.data.DataLoader(cropped_test_set, batch_size=args.batch, shuffle=False) for cropped_test_set in cropped_test_sets]

    # print(f'{[len(cropped_train_set) for cropped_train_set in cropped_train_sets]}')
    # print(f'{[len(cropped_test_set) for cropped_test_set in cropped_test_sets]}')

    shuffled_idxes = [list(range(0, len(test_datasets[idx]))) for idx in range(len(test_datasets))]
    for idx in range(len(shuffled_idxes)):
        random.shuffle(shuffled_idxes[idx])
    concated_test_set = [torch.utils.data.Subset(test_datasets[idx], shuffled_idxes[idx][:min_data_len]) for idx in range(len(test_datasets))]
    concated_test_set = torch.utils.data.ConcatDataset(concated_test_set)
    # print(len(concated_test_set))
    concated_test_loader = torch.utils.data.DataLoader(concated_test_set, batch_size=args.batch, shuffle=False)

    return train_datasets, test_datasets, train_loaders, test_loaders, concated_test_loader, MEANS, STDS#, cropped_train_loaders, cropped_test_loaders