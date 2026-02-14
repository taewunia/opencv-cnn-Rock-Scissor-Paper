import torchvision
from torchvision import transforms

TRAIN_PRE_PROCESS = torchvision.transforms.Compose([
                                      transforms.Resize(
                                          (256, 256)
                                      ),
                                      transforms.RandomAffine(
                                                              degrees=60 ,
                                                              translate=(0.2, 0.2),
                                                              scale=(0.7, 1.2),
                                                              ),
                                      transforms.ColorJitter(0.2,
                                                             0.2),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.5, 0.5, 0.5],
                                          std=[0.5, 0.5, 0.5]
                                      )
                                      ])

TEST_PRE_PROCESS = torchvision.transforms.Compose([
                                      transforms.Resize(
                                          (256, 256)
                                      ),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.5, 0.5, 0.5],
                                          std=[0.5, 0.5, 0.5]
                                      )
])



TRANSFER_PRE_PROCESS = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(
            224, scale=(0.4, 1.0), ratio=(0.75, 1.33)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ]),

    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])
}