import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch
from model.AudioVisualSeparator import AudioVisualSeparator
import pickle
from dataset.MusicDataset import MusicDataset

if __name__ == "__main__":
    mod = AudioVisualSeparator()
    for param in mod.parameters():
        pass
        #print(type(param), param.size())

    file_path = r'C:\Users\user\Desktop\data'
    dataset = MusicDataset(file_path, None, None)
    #
    # try:
    #     mix_file = open(file_path, 'rb')
    #     pick = pickle.load(mix_file)
    #     mix_file.close()
    # except OSError as err:
    #     #self.log.write("-->> Error with file " + file_path)
    #     pick = None
    #     print(err)
    #     exit(-1)
    #
    # X = pick
    # t = X['obj1']['images'][:]
    # print(len([c[1] for c in X['obj1']['images'][:]] + [c[1] for c in X['obj2']['images'][:]]))
    # mix = T.ToTensor()(X['mix'][0])
    # print(mix.shape)
    # #print(T.ToTensor()(mix).shape)
    # k = T.ToTensor()(X['obj1']['images'][0][1]).unsqueeze(0)
    # k = np.vstack(k)
    # print(k.shape)
    # Y = None
    # print("len : " + str(len([[X['obj1']['audio']['stft']], [X['obj2']['audio']['stft']]][0][0])))

    X = dataset.__getitem__(0)
    #print(X)
    Y = mod(X)
    print(Y)




    #path = r"C:\Users\user\Desktop\etc\chunk_10\cropped_000011\39.jpg"

    '''
    im = Image.open(path).resize((224, 224))
    #im.show()

    tim = T.ToTensor()(im)
    #tim.res(3*224*224)
#    print(torch.max(tim[0]))
    tsfm = T.Normalize(mean=(0.1057, 0.1525, 0.1557), std=(0.0953, 0.1483, 0.1151))
    lis = tsfm(tim)
    nor = T.ToPILImage()(lis)
    #nor.show()


    pix = np.asarray(im).astype('float32')
    a = [pix]
    a += [pix]
    a = [np.ones((2, 3))]
    a[0][0][0] = 9
#    a = T.ToTensor()(np.asarray([tim, tim]))
    a = (tim + tim) / 2
    a -= tim
    print(a)
    print("ANS:")
    print(a.std((1, 2)))
    #print(a.mean())
    #print(np.std(a[0][1][:][:]))
    #print(np.std(a[0][:][:][2]))
    #print(a[0][3])
    #print(len(a[0]))
    '''