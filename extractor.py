# coding: utf-8

import sunau
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np 
import struct
import os
import pandas as pd
import simpleaudio as sa
import wave
import pickle


def read_file(name):

    fs = 22050

    wave_read = wave.open(name, 'rb')
    nchannels, sampwidth, framerate, nframes, comptype, compname = wave_read.getparams()
    data = wave_read.readframes(nframes)

    # au = sunau.open(name, 'r')
    # nchannels, sampwidth, framerate, nframes, comptype, compname = au.getparams()
    # data = au.readframes(nframes)
    # print('nchannels : {}, sampwidth : {}, framerate : {}, nframes : {}, comptype : {}, compname : {}'.format(nchannels, sampwidth, framerate, nframes, comptype, compname))
    # wave_obj = sa.WaveObject(data, nchannels, sampwidth, framerate)
    # play_obj = wave_obj.play()
    # play_obj.wait_done()

    # data = ""
    # with open(name, 'rb') as au:
    #     magic = au.read(4)
    #     dataoffset = int.from_bytes(au.read(4), byteorder='big')
    #     datasize = int.from_bytes(au.read(4), byteorder='big')
    #     au.seek(0)
    #     au.read(dataoffset)
    #     data = au.read(datasize)

    

    d = []
    for i in range(0, nframes, 2):
        d.append(struct.unpack('h', data[i:i+2])[0])
    d = np.array(d)

    
    f, t, Zxx = signal.stft(d, fs, nperseg=512)

    plt.plot(d)
    plt.show()
    
    Zxx = np.abs(Zxx)
    means = np.mean(Zxx, axis=1)
    stds = np.std(Zxx, axis=1)
    return means, stds

if __name__ == "__main__":

    if False:
    ############# test ###########
        val = read_file('genres/blues/blues.00000.wav')

    ##############################

    else:

        df = {}

        # récupération des catégories
        cats = []
        for root, dirs, files in os.walk('genres'):
            for d in dirs:
                cats.append(d)

        j = 1
        j_max = len(cats)
        for cat in cats:
            for root, dirs, files in os.walk('genres/'+cat):
                imax = len(files)
                print(cat + ' : ')
                i = 1
                for fi in files:
                    val = read_file('genres/'+cat+'/'+fi)
                    df['{}.{}'.format(cat, i)] = val
                    print('({}/{}).'.format(j, j_max), i, '/', imax, ' -> ', fi)
                    i+=1
            j+=1

        
        pickle.dump(df, open( "data.pkl", "wb" ))


