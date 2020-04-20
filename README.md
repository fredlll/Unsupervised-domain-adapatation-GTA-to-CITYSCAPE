# DeepLabv3Plus-Pytorch

## GTA

### 1. Download 'DeepLabV3Plus-Pytorch' folder 

### 2. Divide GTA into train, val, test parts and put in folder as follow:

```
/datasets
    /data
        /GTA
            /gtFine                    #This folder saves label images
                /train                 #about 70 % images
                    /city
                        /1.jpg
                        /2.jpg
                        ......
                /test                  #about 70 % images
                    /city
                        /*.jpg
                /val                   #about 70 % images
                    /city
                        /*.jpg
            /leftImg8bit               #This folder saves original images
                /train                 #corresponding to gtFine
                    /city
                        /*.jpg
                /test
                    /city
                        /*.jpg
                /val
                    /city
                        /*.jpg
```

### 3. Train your model on GTA
```
If you want to retrain from the beginning, just run main.py

If you want to load from pretrained or checkpoint path:

parser.add_argument("--ckpt", default='checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth', type=str,
                        help="restore from checkpoint")

set default to the name of your pth, or run:

python main.py ... --ckpt YOUR_CKPT --continue_training

Also you can set another important parameters in parser.add_argument:


# save result images by setting default to True or run python main.py ... --save_val_results True
parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
# followings are the same operations
parser.add_argument("--test_only", action='store_true', default=False)
parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.01)")
parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')

```

