# import torchvision
# import torchvision.transforms as transforms

import requests
import os 

def download_usps():
    # url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2'
    # r = requests.get(url, allow_redirects=True)
    print(os.getcwd())
    if not os.path.isdir('../../data/raw/USPS/'):
        os.mkdir('../../data/raw/USPS/')
    open('USPS/usps.bz2', 'wb').write(r.content)
