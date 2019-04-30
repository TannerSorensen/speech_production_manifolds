#!/usr/bin/env python

from sklearn.model_selection import train_test_split
from scipy import misc
import pandas as pd
import numpy as np
import json
from glob import glob
import os
import sys

def label(f,df):
    spkr, _, avi, filename = f.split(os.path.sep)[-4:]
    if avi in df.avi.values:
        img = float(filename.replace('.png',''))
        phone = df.loc[df[(df.avi == avi) & (df.img == img)].index,'phone'].values
        if phone.shape[0] > 0:
            phone = phone[0]
        else:
            phone = None
    else:
        phone = None

    sys.stdout.write('\r'+f)

    return phone, spkr

def read_nsf_vtsf(filename):
    with np.load(filename) as f:
        return f['x_train'], f['x_val'], f['x_test'], \
            f['phone_train'], f['phone_val'], f['phone_test'], \
            f['spkr_train'], f['spkr_val'], f['spkr_test'], \
            f['align_train'], f['align_val'], f['align_test'], \
            f['fname_train'], f['fname_val'], f['fname_test']

def label_phones(filenames, align_dir):
    tr_per_img = 2.0
    tr = 0.006004
    fs = 1.0 / (tr_per_img * tr)

    def sec2frame(t):
        s = int(np.round(t * fs))
        return s

    phone_list = [None] * len(filenames)
    avi_list = [ f.split(os.path.sep)[-2] for f in filenames ]
    img_list = [ int(f.split(os.path.sep)[-1].replace('.png','')) for f in filenames ]

    # list json files
    json_files = glob(os.path.join(align_dir,'*','align','*.json'))

    for json_idx, json_filename in enumerate(json_files):
        sys.stdout.write('\r'+str(json_idx+1)+'/'+str(len(json_files))+' '+json_filename)

        # load the json file
        d = json.loads(open(json_filename).read())

        # determine which avi file the json file corresponds to
        avi = json_filename.split(os.path.sep)[-1].replace('.json','')

        for w in d['words']:
            if w['case']=='success':
                w_name = w['alignedWord']

                # get start time for word
                word_start_time = float(w['start'])

                # offset tracks time since start of word
                offset = 0.0
                for ph in w['phones']:
                    ph_name = ph['phone']

                    # get onset and end of phone in seconds
                    ons = word_start_time + offset
                    offset += float(ph['duration'])
                    end =  ons + offset

                    # convert onset and end of phone to frames
                    ons = sec2frame(ons)
                    end = sec2frame(end)

                    # set phone for all images in range
                    for idx, (a, img) in enumerate(zip(avi_list, img_list)):
                        if (a == avi) and (img >= ons) and (img <= end):
                            phone_list[idx] = ph_name.split('_')[0]

    return np.array(phone_list)

if __name__=="__main__":

    data_path = sys.argv[1]
    align_path = sys.argv[2]

    # get list of filenames
    fname = glob(os.path.join(data_path,'*','png','*','*.png'))

    # get list of images
    print('Loading image data')
    img_list = [None] * len(fname)
    for idx, f in enumerate(fname):
        sys.stdout.write('\r'+str(idx+1)+'/'+str(len(fname))+' '+f)
        img_list[idx] = misc.imread(f)
    img = np.array(img_list)
    print('')

    print('Convert images to float32, normalize to range [0,1], reshape to size (?,84,84,1)')
    img = np.float32(img) / img.max()
    img = np.reshape(img, img.shape+(1,))

    # read manual annotations
    df = pd.read_csv(os.path.join('annotations','endpoints.csv'))

    print('Labeling phones (manual annotation labels)')
    phone, spkr = map(list, zip(*[label(f,df) for f in fname]))
    print('')

    print('Labeling phones (gentle forced alignment labels)')
    align = label_phones(fname, align_path)

    # train/val/test split 0.5/0.25/0.25
    print('\nGenerate train/val/test split (50/25/25)')
    x_train, x_test, fname_train, fname_test, phone_train, phone_test, spkr_train, spkr_test, align_train, align_test \
        = train_test_split(img, fname, phone, spkr, align, test_size=0.25, stratify=spkr)
    x_train, x_val, fname_train, fname_val, phone_train, phone_val, spkr_train, spkr_val, align_train, align_val \
        = train_test_split(x_train, fname_train, phone_train, spkr_train, align_train, test_size=0.33, stratify=spkr_train)

    print('Saving to disk')
    np.savez_compressed('nsf_vtsf.npz', \
        x_train=x_train, x_val=x_val, x_test=x_test, \
        phone_train=phone_train, phone_val=phone_val, phone_test=phone_test, \
        spkr_train=spkr_train, spkr_val=spkr_val, spkr_test=spkr_test, \
        align_train=align_train, align_val=align_val, align_test=align_test, \
        fname_train=fname_train, fname_val=fname_val, fname_test=fname_test)
