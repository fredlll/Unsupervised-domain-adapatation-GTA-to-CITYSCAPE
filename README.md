

## Unsupervised-domain-adapatation-GTA-to-CITYSCAPE

### 1. git clone git@github.com:fredlll/Unsupervised-domain-adapatation-GTA-to-CITYSCAPE.git
		cd Unsupervised-domain-adapatation-GTA-to-CITYSCAPE
		pip -r requirements.txt 

### 2. Divide GTA into train, val, test parts and put in folder as follow:

```
/datasets
    /data
        /images                  # gta-as-cityscape images
            /1.jpg
            /2.jpg
                        ......
        /labels                 # gta-as-cityscape labels
            /1.jpg
            /2.jpg
                        ......        
        /Cityscape                
            /images                  # cityscape images
                        /1.jpg
                        /2.jpg
                        ......
            /labels                 # cityscape labels
                        /1.jpg
                        /2.jpg
```

### 3. 1st step, Train your model on GTA -as-cityscape
```
python main_GTAasC.py ...

```

### 4. 2nd step, Train domain adaptation using GTA -as-cityscapepretrain weight
```
sh train.sh

```