#!/usr/bin/env python

from sklearn.model_selection import train_test_split
from scipy import misc
import pandas as pd
import numpy as np
from glob import glob
import os
import sys

def label(f,df):
    spkr, _, avi, filename = f.split(os.path.sep)[-4:]
    if avi in df.avi.values:
        img = float(filename.replace('.png',''))
        phone = df.loc[df[(df.avi == avi) & (df.img == float(img))].index,'phone'].values
        if phone.shape[0] > 0:
            phone = phone[0]
        else:
            phone = None
    else:
        phone = None

    sys.stdout.write('\r'+f)

    return phone, spkr
    

if __name__=="__main__":
    # get list of filenames
    fname = glob(os.path.join('data','F101','png','*','*.png'))

    # get list of images
    print('Loading image data')
    img_list = [None] * len(fname)
    for idx, f in enumerate(fname):
        sys.stdout.write('\r'+str(idx+1)+'/'+str(len(fname))+' '+f)
        img_list[idx] = misc.imread(f)
    img = np.array(img_list)
    print('')

    print('Convert images to float32, normalize to range [0,1], reshape to size (?,84,84,1)')
    mx = np.max(img, axis=0)
    img = img.astype('float32') / mx
    img = np.reshape(img, img.shape+(1,))

    # read manual annotations
    df = pd.read_csv(os.path.join('annotations','endpoints.csv'))

    print('Labeling phones')
    phone, spkr = map(list, zip(*[label(f,df) for f in fname]))
    
    # train/val/test split 0.5/0.25/0.25
    print('\nGenerate train/val/test split (50/25/25)')
    x_train, x_test, fname_train, fname_test, phone_train, phone_test, spkr_train, spkr_test \
        = train_test_split(img, fname, phone, spkr, test_size=0.25, stratify=spkr)
    x_train, x_val, fname_train, fname_val, phone_train, phone_val, spkr_train, spkr_val \
        = train_test_split(x_train, fname_train, phone_train, spkr_train, test_size=0.33, stratify=spkr_train)


    print('Saving to disk')
    np.savez('nsf_vtsf.npz', \
        x_train=x_train, x_val=x_val, x_test=x_test, \
        phone_train=phone_train, phone_val=phone_val, phone_test=phone_test, \
        spkr_train=spkr_train, spkr_val=spkr_val, spkr_test=spkr_test, \
        fname_train=fname_train, fname_val=fname_val, fname_test=fname_test)
