
import argparse
import matplotlib.pyplot as plt
import tempfile
import torch
import torchaudio
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import librosa
#import IPython
from scipy.io.wavfile import write
import numpy as np
import soundfile
import torch.nn.functional as F
from PIL import Image
import os

def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid

def plot_spectrogram(spec, id, round, dir, name, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec.cpu()), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)
    im_name = '[round ' + str(round) + '] - test_spectrogram_' + name + "_vid_id_" + id + '.png'
    path = r'/home/dsi/ravivme/run-model/Servers-DeepProject/DeepProject-Servers/specs/'
    plt.savefig(path + im_name)
    plt.savefig(dir + im_name)
    
def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=8,
        shuffle=True,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.device(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    round = 0

    with torch.no_grad():
        for i, pick in enumerate(tqdm(data_loader)):
            round += 1

            root = f"/dsi/gannot-lab/datasets/Music/Graphs/"
            dir = os.path.join(root, "round_" + str(round))
            try:
                os.mkdir(dir)
            except OSError:
                pass

            print("->" + str(i) + "\n")
            #data, target = data.to(device), target.to(device)
            #output = model(data)
            
            model.to(device)
            bs = pick["detections"].shape[0] * pick["detections"].shape[1]
            
            pick['detections'] = pick['detections'].view(bs, 3, 224, 224)
            
            for key, _ in pick.items():
                pick[key] = pick[key].to(device)

            #model.zero_grad()
            output = model(pick)

            ground_masks = output["ground_masks"]#.numpy()
            predicted_masks = output["predicted_masks"]#.clone()#.detach().numpy()
            weights = output["weights"]
            labels = output["ground_labels"]#.view(-1)#.numpy()   #.astype(np.float32)
            
            pred_labels= output['predicted_audio_labels']#.clone()#.detach().numpy()  #.view(-1).numpy()
            
            vec = torch.ones(predicted_masks.shape)

            for idx in range(len(labels)-1, -1, -2):  
                 if labels[idx] == 15:
                    vec[idx] = torch.zeros(vec[idx].shape)

            vec = vec.to(device)
            weights = weights.view((int(bs / 2), 2, 1, 256, 256))
            loss = loss_fn((predicted_masks * vec).view(int(bs / 2), 2, 1, 256, 256), ground_masks.view(int(bs / 2), 2, 1, 256, 256), weights)
            
            i = 4
            vid = output['videos']
            im = output['detections'][i].T.detach().cpu().numpy()
            print(output['detections'].shape)
            print(";;;;;;;;;;;;;;;")
#            im = pick['obj2']['images'][0][1] / 255
            #im = im[:,:,::-1]
            print(im)

            plt.imshow(im)# / (2550))
            plt.show()
            plt.savefig('./detect1.png')

            plt.figure()
            im = output['detections'][i+1].T.detach().cpu().numpy()
            Image.fromarray((im * 255).astype(np.uint8), 'RGB').save("./trial.jpg")
            plt.imshow(im)# / (2550))
            plt.show()
            plt.savefig('./detect2.png')

            plt.figure()
            im = output['detections'][i+2].T.detach().cpu().numpy()
            plt.imshow(im)# / (2550))
            plt.show()
            plt.savefig('./detect3.png')

            plt.figure()
            im = output['detections'][i+3].T.detach().cpu().numpy()
            plt.imshow(im)# / (2550))
            plt.show()
            plt.savefig('./detect4.png')


            # spectrograms
            audio_mags = output['audio_mags'] #.view(bs, 1, 512, 256)            #['stft'][0], X['obj2']['audio']['stft'][0]]  #array includes both videos data - 2 values
            #ground_mask = pick["ground_masks"]
            mixed_audio = output["mixed_audios"]
            vid_ids = output["videos"]

            plot_spectrogram(audio_mags[0][0], "audio_" + str(vid_ids[0]), round)
            plot_spectrogram(audio_mags[2][0], "audio_" + str(vid_ids[2]), round)
            plot_spectrogram(ground_masks[0][0] + ground_masks[1][0], "mask_" + str(vid_ids[0]), round)
            plot_spectrogram(ground_masks[2][0] + ground_masks[3][0], "mask_" + str(vid_ids[2]), round)
            plot_spectrogram(mixed_audio[0][0], "mix_" + str(vid_ids[0]) + " & " + str(vid_ids[2]), round)
            
            # for i, mask in enumerate(ground_mask):
            #     plot_spectrogram(mask[0], str(vid_ids[i]))


            print("vid id : " + str(vid))
            print(output['audio_phases'].shape)
            print(output['original_mixed_audio'].shape)
            print("???????????????")
            rate = pick['rate']
            audio = output['predicted_spectrograms'][i]#.view(-1)
            print(audio.shape)
            print(rate)
            
            phase_mix = output['audio_phases'][i][0].squeeze().detach().cpu().numpy()
            mag_mix = output['original_mixed_audio'][i][0].squeeze().detach().cpu().numpy()
            
            print("**************")
            print(mag_mix.shape)
            print(phase_mix.shape)
            print("**************")
            B = bs#mag_mix.shape[0]
            stft_frame = 1022
            pred_masks_ = output['predicted_masks']
            grid_unwarp = torch.from_numpy(warpgrid(B, stft_frame//2+1, pred_masks_.size(3), warp=False)).to(device)
            pred_masks_linear = F.grid_sample(pred_masks_, grid_unwarp)

            print(pred_masks_linear[0, 0])
            print(pred_masks_linear[1, 0])
            print(pred_masks_linear[2, 0])
            print(pred_masks_linear[3, 0])
            # convert into numpy
            #mag_mix = mag_mix.numpy()
            #phase_mix = phase_mix.numpy()
            pred_masks_linear = pred_masks_linear#.detach().cpu().numpy()
            pred_mag = mag_mix * pred_masks_linear[4*i, 0].cpu().numpy()

            pred_mag1 = mag_mix * pred_masks_linear[4*i+1, 0].cpu().numpy()
            pred_mag2 = mag_mix * pred_masks_linear[4*i+2, 0].cpu().numpy()
            pred_mag3 = mag_mix * pred_masks_linear[4*i+3, 0].cpu().numpy()

            print("!!!**************!!!")
            print(pred_mag + pred_mag1)
            print(np.sum(pred_mag1))
            print("!!!**************!!!")

            
            print("<<<**************>>>")
            print(mag_mix.shape)
            print(pred_masks_linear.shape)
            print("<<<**************>>>")




            spec = pred_mag.astype(np.complex) * np.exp(1j*phase_mix)
            
            #spec = pred_mag.detach().cpu().numpy().astype(np.complex) * np.exp(1j*phase_mix)
            
            audio = librosa.istft(spec, hop_length=256, center=True, length=65535)#.tolist()#, rate)
            audio = np.clip(audio, -1., 1.)        

            path = f"/dsi/gannot-lab/datasets/Music/saved_example.wav"
            soundfile.write(path, audio.T, 11025, format='wav')
            




            spec = pred_mag1.astype(np.complex) * np.exp(1j*phase_mix)
            
            #spec = pred_mag.detach().cpu().numpy().astype(np.complex) * np.exp(1j*phase_mix)
            
            audio = librosa.istft(spec, hop_length=256, center=True, length=65535)#.tolist()#, rate)
            audio = np.clip(audio, -1., 1.)        

            path = f"/dsi/gannot-lab/datasets/Music/saved_example1.wav"
            soundfile.write(path, audio.T, 11025, format='wav')


            


            spec = pred_mag2.astype(np.complex) * np.exp(1j*phase_mix)
            
            #spec = pred_mag.detach().cpu().numpy().astype(np.complex) * np.exp(1j*phase_mix)
            
            audio = librosa.istft(spec, hop_length=256, center=True, length=65535)#.tolist()#, rate)
            audio = np.clip(audio, -1., 1.)        

            path = f"/dsi/gannot-lab/datasets/Music/saved_example2.wav"
            soundfile.write(path, audio.T, 11025, format='wav')

            


            spec = pred_mag3.astype(np.complex) * np.exp(1j*phase_mix)
            
            #spec = pred_mag.detach().cpu().numpy().astype(np.complex) * np.exp(1j*phase_mix)
            
            audio = librosa.istft(spec, hop_length=256, center=True, length=65535)#.tolist()#, rate)
            audio = np.clip(audio, -1., 1.)        

            path = f"/dsi/gannot-lab/datasets/Music/saved_example3.wav"
            soundfile.write(path, audio.T, 11025, format='wav')


#            IPython.display.Audio(path)

            '''
            with tempfile.TemporaryDirectory() as tempdir:
                path = f"/dsi/gannot-lab/datasets/Music/saved_example.mp3"
                #librosa.output.write_wav(path, audio, rate)
                soundfile.write(path, audio, 11025, "PCM_24")
#                write(path, rate, audio.astype(np.float32))
            '''

            
            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            
            #loss = loss_fn(output, target)
            
            #batch_size = data.shape[0]
            batch_size = bs
            total_loss += loss.item() #* batch_size / 2
            print(total_loss)
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(pred_labels, labels) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)