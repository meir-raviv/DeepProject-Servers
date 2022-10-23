from nntplib import GroupInfo
from matplotlib.afm import CompositePart
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
from base.base_trainer import BaseTrainer
from utils import inf_loop, MetricTracker
from model import loss
import matplotlib.pyplot as plt
import librosa
from torch import Tensor
#import torchvision.transforms as T
import torch.nn.functional as F
from torchvision import transforms as T
from asteroid import filterbanks
from asteroid_filterbanks.enc_dec import Filterbank, Encoder, Decoder
from asteroid_filterbanks import FreeFB
import asteroid_filterbanks



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


def plot_spectrogram(pick, ind, p, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  spec = pick['obj2']['audio']['stft'][p][0][0]
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)
  plt.savefig('./spec_new/spec'+str(ind)+'.png')


def _check_same_shape(preds: Tensor, target: Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError("Predictions and targets are expected to have the same shape")

def scale_invariant_signal_distortion_ratio(preds: Tensor, target: Tensor, zero_mean: bool = True) -> Tensor:
    """Calculates Scale-invariant signal-to-distortion ratio (SI-SDR) metric. The SI-SDR value is in general
    considered an overall measure of how good a source sound.
    Args:
        preds:
            shape ``[...,time]``
        target:
            shape ``[...,time]``
        zero_mean:
            If to zero mean target and preds or not
    Returns:
        si-sdr value of shape [...]
    Example:
        #>>> from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
        #>>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        #>>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        #>>> scale_invariant_signal_distortion_ratio(preds, target)
        tensor(18.4030)
    References:
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP) 2019.
    """
    #print(f"shape preds: {preds.shape} \nshape target: {target.shape}")
    _check_same_shape(preds, target)
    EPS = torch.finfo(preds.dtype).eps

    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)

    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + EPS) / (
        torch.sum(target ** 2, dim=-1, keepdim=True) + EPS
    )
    target_scaled = alpha * target

    noise = target_scaled - preds

    val = (torch.sum(target_scaled ** 2, dim=-1) + EPS) / (torch.sum(noise ** 2, dim=-1) + EPS)
    val = 10 * torch.log10(val)
    # print(val.shape)
    return -val

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config

        self.model.device(device)

        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch, lost_loss, t, ind, lost_loss_train, t_train, ind_train):
        print("start train epoch")
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        lost_loss1 = []

        for batch_idx, pick in enumerate(self.data_loader):
            
            self.model.to(self.model.device)
            # num_of_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            # print("Number of parameters that require grad in the model is: {num}".format(num=num_of_param))
            
            #data, target = pick.to(self.device), None#, target.to(self.device)
            #pick['mixed_audio'] = pick['mixed_audio'].view(4, 15, 512, 256)
            #pick['mixed_audio'] = pick['mixed_audio'].to(self.device)
            bs = pick["detections"].shape[0] * pick["detections"].shape[1]
            
            # print("detections before")
            # print(pick['detections'].shape)
            pick['detections'] = pick['detections'].view(bs, 3, 224, 224)
            # print(pick['detections'].shape)
            # print("detections before")

            #pick['detections'] = pick['detections'].to(self.device)
            #pick['classes'] = pick['classes'].to(self.device)
            
            for key, value in pick.items():
                pick[key] = pick[key].to(self.model.device)

            for p in range(4):
                #plot_spectrogram(pick, ind[0], p)
                pass

            self.model.zero_grad()
            output = self.model(pick)

            ground_masks = output["ground_masks"]#.numpy()
            #print(ground_masks.shape)
            #ground_masks = ground_masks.view(32, ground_masks.shape[2], ground_masks.shape[3])
            predicted_masks = output["predicted_masks"]#.clone()#.detach().numpy()
            #print(predicted_spectrograms.shape)
            #predicted_spectrograms = predicted_spectrograms.view(32, predicted_spectrograms.shape[2], predicted_spectrograms.shape[3])
            weights = output["weights"]
            #print(weights.shape)
            #weights = weights.view(bs, 1, weights.shape[2], weights.shape[3])#.numpy()

            #start from 0?
            labels = output["ground_labels"]#.view(-1)#.numpy()   #.astype(np.float32)
            
            pred_labels= output['predicted_audio_labels']#.clone()#.detach().numpy()  #.view(-1).numpy()
            #pred_labels = np.zeros(32)
            
            #for idx in range(len(predicted_labels)-1, -1, -1):
            #   pred_labels[idx] = np.argmax(predicted_labels[idx])
            
            #print(ground_masks[0])

            '''
            for idx in range(len(labels)-1, -1, -2):
                ground_masks = np.delete(ground_masks, idx, axis=0)
                if labels[idx] != -2:
                    predicted_masks[idx - 1] += predicted_masks[idx]        
                    weights = np.delete(weights, idx, axis=0)
                    pred_labels = np.delete(pred_labels, idx, axis=0)
                    labels = np.delete(labels, idx, axis=0)        
                predicted_masks = np.delete(predicted_masks, idx, axis=0)
            '''



            # for idx in range(len(labels)-1, -1, -1):
            #     #print(labels[idx])
            #     if labels[idx] == -2:
            #         #print(idx)
            #         ground_masks = np.delete(ground_masks, idx, axis=0)
            #         predicted_masks = np.delete(predicted_masks, idx, axis=0)
            #         weights = np.delete(weights, idx, axis=0)
            #         pred_labels = np.delete(pred_labels, idx, axis=0)
            #         labels = np.delete(labels, idx, axis=0)
            
            #print("-------")
            #print(pred_labels.shape)
            
            '''
            for idx in range(len(predicted_masks)):
#                predicted_spectrograms[idx] = torch.clamp(torch.from_numpy(predicted_spectrograms[idx]), 0, 1)
                predicted_masks[idx] = np.clip(predicted_masks[idx], 0, 1)
                ground_masks[idx] = np.clip(ground_masks[idx], 0, 1)
            '''



                #should we clamp input also>?????
            
            '''
            weights = np.expand_dims(weights, axis=1)
            ground_masks = torch.from_numpy(ground_masks)
            '''
            #torch.unsqueeze(weights, 1)

            #??????????????
            #predicted_spectrograms = predicted_spectrograms[:, 0, :, :]
            #predicted_spectrograms = np.expand_dims(predicted_spectrograms, axis = 1)
            
            '''
            predicted_masks = torch.from_numpy(predicted_masks)#.squeeze(axis=1)
            
            weights = torch.from_numpy(weights)

            pred_labels = torch.from_numpy(pred_labels)
            labels = torch.from_numpy(labels)
            '''



            # for i in range(len(labels)):
            #     print("ground : " + str(labels[i]))
            #     n = -1
            #     jj = -1
            #     for j, k in enumerate(pred_labels[i]):
            #         if k > n:
            #             n = k
            #             jj = j
            #     print("pred : " + str(jj))
            #     print()
            # print("next")

            
            #print(ground_masks.shape)
            #print(predicted_spectrograms.shape)
            #print(weights.shape)

            # should we mul with weights the result of loss?
            
            #coseparation_loss = 0
            
            # print("weights before")
            # print(weights.shape)
            weights = weights.view((int(bs / 2), 2, 1, 256, 256))
            # print(weights.shape)
            # print("weights after")

            # print("pred + masks before")
            # print((predicted_masks * vec).shape)
            # print(ground_masks.shape)
            
            # coseparation_loss = self.criterion((predicted_masks * vec).view(int(bs / 2), 2, 1, 256, 256), ground_masks.view(int(bs / 2), 2, 1, 256, 256), weights)
            
            # print("pred + masks after")
            # print((predicted_masks * vec).view(int(bs / 2), 2, 1, 256, 256).shape)
            # print(ground_masks.view(int(bs / 2), 2, 1, 256, 256).shape)

            #coseparation_loss = 0

            # print("****audio phases shape****")
            # print(output["audio_phases"].shape)



            phase_mix = output['audio_phases'].view(int(bs/2), 2, 512, 256)#.detach().cpu().numpy()     # [:, 0]
            mag_mix = output['original_mixed_audio'].view(int(bs/2), 2, 512, 256)#.detach().cpu().numpy()     # [:, 0]
            B = bs#mag_mix.shape[0]
            stft_frame = 1022
            pred_masks_ = output['predicted_masks']
            grid_unwarp = torch.from_numpy(warpgrid(B, stft_frame//2+1, pred_masks_.size(3), warp=False)).to(self.model.device)
            pred_masks_linear = F.grid_sample(pred_masks_, grid_unwarp).view(int(bs/2), 2, 512, 256)
            
            
            pred_mag = mag_mix * pred_masks_linear#.clone().detach().cpu().numpy() # [:, 0]

            # print("pred_mag shape")
            # print(pred_mag.shape)
            
            # spec = pred_mag.astype(np.complex) * np.exp(1j*phase_mix)
            spec = asteroid_filterbanks.transforms.from_magphase(pred_mag, phase_mix, dim = -2)

            vec = torch.ones(32, 1, 1024, 256)

            for idx in range(len(labels)-1, -1, -2):  
                 if labels[idx] == 15:
                    vec[idx] = torch.zeros(vec[idx].shape)
            
            vec = vec.to(self.model.device)
            vec = vec.view(16, 2, 1024, 256)
            spec = spec * vec
            # print(spec)
            spec = spec.view(16, 2, 2, 512, 256)
            spec = torch.sum(spec, axis=1)#.view(16, 512, 256, 2)
            spec = torch.swapaxes(spec, 1, 3)
            spec = torch.swapaxes(spec, 1, 2)


                    

            # a = torch.randn(6)
            # print(a)
            # print(a.view(2, 3))

            # #spec = torch.view_as_real(spec)
            # print(spec.shape)

            # spec = spec.view(bs/2, 2, 512, 256)
            audio = torch.istft(spec, hop_length=256, center=True, length=65535, n_fft=1022)              #.tolist()#, rate)
            # print(audio.shape)

            pred_audio = torch.clamp(audio, -1., 1.)





            #phase_mix = output['audio_phases'].squeeze().detach().cpu().numpy()     # [:, 0]
            #mag_mix = output['original_mixed_audio'].view(32, 1, 512, 256).squeeze().detach().cpu().numpy()     # [:, 0]
            #B = bs#mag_mix.shape[0]
            #stft_frame = 1022
            ground_masks_ = output['ground_masks']
            grid_unwarp = torch.from_numpy(warpgrid(B, stft_frame//2+1, ground_masks_.size(3), warp=False)).to(self.model.device)
            ground_masks_linear = F.grid_sample(ground_masks_, grid_unwarp).view(int(bs/2), 2, 512, 256)

            ground_mag = mag_mix * ground_masks_linear#.cpu().numpy() # [:, 0]

            # spec = ground_mag.astype(np.complex) * np.exp(1j*phase_mix)
            spec = asteroid_filterbanks.transforms.from_magphase(ground_mag, phase_mix, dim = -2).view(16, 2, 2, 512, 256)
            spec = spec[:, 0, :, :, :]
            spec = torch.swapaxes(spec, 1, 3)
            spec = torch.swapaxes(spec, 1, 2)

            
            # ground_audio = librosa.istft(spec, hop_length=256, center=True, length=65535)#.tolist()#, rate)
            # ground_audio = np.clip(ground_audio, -1., 1.)
            
            ground_audio = torch.istft(spec, hop_length=256, center=True, length=65535, n_fft=1022)              #.tolist()#, rate)
            # print(audio.shape)

            ground_audio = torch.clamp(ground_audio, -1., 1.)
            
            
            
            # print("pred audio shape:")
            # print(pred_audio.shape)
            # print("ground audio shape:")
            # print(ground_audio.shape)

            # print("ground audio shape:")
            # print(output["ground_audios"].shape)



            #ground_audio = output["ground_audios"].view(int(bs/2), 2, 512, 256)

            #ground_audio = ground_audio[:, 0, :]

            # print("ground audio np shape:")
            # print(ground_audio.shape)

            #pred_audio = np.sum(pred_audio, axis=1)
            
            # print("pred audio np shape:")
            # print(pred_audio.shape)

            coseparation_loss = scale_invariant_signal_distortion_ratio(preds=pred_audio, target=ground_audio)

            '''
            coseparation_loss = 0

            for idx in range(len(labels)-1, -1, -2):
                if labels[idx] == 15:
                    #predicted_masks[idx] = predicted_masks[idx] * 0
                    coseparation_loss += self.criterion(predicted_masks[idx - 1], ground_masks[idx])#, weights)
                else:
                    coseparation_loss += self.criterion(predicted_masks[idx - 1] + predicted_masks[idx], ground_masks[idx])#, weights)
            #coseparation_loss = torch.FloatTensor(coseparation_loss)
            #coseparation_loss = coseparation_loss /32

            coseparation_loss = torch.mean(coseparation_loss)
            '''
            #print("-------")
            #print(pred_labels.shape)
            #print(labels.shape)

            lamda = 0.01
            #consistency_loss = loss.ce_loss(pred_labels, Variable(labels, requires_grad=False)) * lamda
            coseparation_loss = coseparation_loss.mean()
            sum_loss = coseparation_loss #+ consistency_loss
            
            self.optimizer.zero_grad()
            #consistency_loss.backward(retain_graph=True)
            coseparation_loss.backward()


            lost_loss1 += [coseparation_loss.cpu().detach().numpy()]

            # lost_loss += [coseparation_loss.cpu().detach().numpy()]
            # t += [ind[0]]
            # ind[0] += 1

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', sum_loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(pred_labels, labels))
                #add mags   self.train_metrics.update(met.__name__, met(labels, pred_labels))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    sum_loss.item()))
                self.writer.add_image('input', make_grid(pick['detections'].cpu(), nrow=8, normalize=True))
            

            # fig, axs = plt.subplots(1, 1)
            # axs.set_title("Loss - epoch: " + str(epoch))
            # axs.set_ylabel('loss')
            # axs.set_xlabel('times')
            
            # #cop = np.asarray(lost_loss)
            # plt.plot(t, lost_loss)
            # #plt.label("epoch: " + str(epoch))
            # plt.show()
            # plt.savefig('loss.png')
            # np.save('loss', lost_loss)
            
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()


        lost_loss_train += [np.mean(lost_loss1)]
        t_train += [ind_train[0]]
        ind_train[0] += 1
        
        fig, axs = plt.subplots(1, 1)
        axs.set_title("Loss - epoch: " + str(epoch))
        axs.set_ylabel('loss')
        axs.set_xlabel('epochs')
        
        #cop = np.asarray(lost_loss)
        plt.plot(t_train, lost_loss_train)
        #plt.label("epoch: " + str(epoch))
        plt.show()
        plt.savefig('train_loss.png')
        np.save('train_loss', lost_loss_train)


        if self.do_validation:
            val_log = self._valid_epoch(epoch, lost_loss, t, ind)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        print("done train epoch")

        # fig, axs = plt.subplots(1, 1)
        # axs.set_title("Loss - epoch: " + str(epoch))
        # axs.set_ylabel('loss')
        # axs.set_xlabel('times')
        
        # #cop = np.asarray(lost_loss)
        # plt.plot(t, lost_loss)
        # #plt.label("epoch: " + str(epoch))
        # plt.show()
        # plt.savefig('loss.png')
        # np.save('loss', lost_loss)

        return log

    def _valid_epoch(self, epoch, lost_loss, t, ind):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        
        lost_loss1 = []

        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, pick in enumerate(self.valid_data_loader):
                
                self.model.to(self.model.device)
            
                #pick['mixed_audio'] = pick['mixed_audio'].to(self.device)
                bs = pick["detections"].shape[0] * pick["detections"].shape[1]
                pick['detections'] = pick['detections'].view(bs, 3, 224, 224)
                #pick['detections'] = pick['detections'].to(self.device)
                #pick['classes'] = pick['classes'].to(self.device)

                for key, value in pick.items():
                    pick[key] = pick[key].to(self.model.device)

                output = self.model(pick)

                ground_masks = output["ground_masks"]#.numpy()
                predicted_masks = output["predicted_masks"]#.clone().detach().numpy()
                weights = output["weights"]
                #weights = weights.view(bs, weights.shape[2], weights.shape[3]).numpy()

                labels = output["ground_labels"].view(-1)#.numpy()   #.astype(np.float32)
                
                pred_labels= output['predicted_audio_labels']#.clone().detach().numpy()  #.view(-1).numpy()
                

                '''
                for idx in range(len(labels)-1, -1, -1):
                    if labels[idx] == -2:
                        ground_masks = np.delete(ground_masks, idx, axis=0)
                        predicted_masks = np.delete(predicted_masks, idx, axis=0)
                        weights = np.delete(weights, idx, axis=0)
                        pred_labels = np.delete(pred_labels, idx, axis=0)
                        labels = np.delete(labels, idx, axis=0)
                
                for idx in range(len(predicted_masks)):
                    predicted_masks[idx] = np.clip(predicted_masks[idx], 0, 1)
                    ground_masks[idx] = np.clip(ground_masks[idx], 0, 1)
                
                weights = np.expand_dims(weights, axis=1)
                ground_masks = torch.from_numpy(ground_masks)

                predicted_masks = predicted_masks[:, 0, :, :]
                predicted_masks = np.expand_dims(predicted_masks, axis = 1)
                predicted_masks = torch.from_numpy(predicted_masks)
                
                weights = torch.from_numpy(weights)

                pred_labels = torch.from_numpy(pred_labels)
                labels = torch.from_numpy(labels)
                '''

                #coseparation_loss = self.criterion(predicted_masks, ground_masks)
                
                # coseparation_loss = 0
                # for idx in range(len(labels)-1, -1, -2):
                #     if labels[idx] == 15:
                #         #predicted_masks[idx] = predicted_masks[idx] * 0
                #         coseparation_loss += self.criterion(predicted_masks[idx - 1], ground_masks[idx])#, weights)
                #     else:
                #         coseparation_loss += self.criterion(predicted_masks[idx - 1] + predicted_masks[idx], ground_masks[idx])#, weights)
                # #coseparation_loss = torch.FloatTensor(coseparation_loss)
                # coseparation_loss = coseparation_loss / 32
                
                # vec = torch.ones(predicted_masks.shape)

                # for idx in range(len(labels)-1, -1, -2):                
                #     vec[idx] = torch.zeros(vec[idx].shape)
                    
                # vec = torch.ones(predicted_masks.shape)

                # for idx in range(len(labels)-1, -1, -2): 
                #      if labels[idx] == 15:
                #         vec[idx] = torch.zeros(vec[idx].shape)

                # vec = vec.to(self.model.device)

                # #vec = vec.view(int(bs / 2), 2)    

                # coseparation_loss = self.criterion((predicted_masks * vec).view(int(bs / 2), 2, 1, 256, 256), (ground_masks * vec).view(int(bs / 2), 2, 1, 256, 256))
                vec = torch.ones(predicted_masks.shape)

                # for idx in range(len(labels)-1, -1, -2):  
                #     if labels[idx] == 15:
                #         vec[idx] = torch.zeros(vec[idx].shape)
                    #if labels[idx] != 15:
                        #predicted_masks[idx-1] += predicted_masks[idx]
                
                vec = vec.to(self.model.device)
                weights = weights.view((int(bs / 2), 2, 1, 256, 256))
                #coseparation_loss = self.criterion((predicted_masks * vec).view(int(bs / 2), 2, 1, 256, 256), (ground_masks * vec).view(int(bs / 2), 2, 1, 256, 256), weights)
                
                phase_mix = output['audio_phases'].view(int(bs/2), 2, 512, 256)#.detach().cpu().numpy()     # [:, 0]
                mag_mix = output['original_mixed_audio'].view(int(bs/2), 2, 512, 256)#.detach().cpu().numpy()     # [:, 0]
                B = bs#mag_mix.shape[0]
                stft_frame = 1022
                pred_masks_ = output['predicted_masks']
                grid_unwarp = torch.from_numpy(warpgrid(B, stft_frame//2+1, pred_masks_.size(3), warp=False)).to(self.model.device)
                pred_masks_linear = F.grid_sample(pred_masks_, grid_unwarp).view(int(bs/2), 2, 512, 256)
                
                pred_mag = mag_mix * pred_masks_linear#.clone().detach().cpu().numpy() # [:, 0]

                spec = asteroid_filterbanks.transforms.from_magphase(pred_mag, phase_mix, dim = -2).view(16, 2, 2, 512, 256)
                spec = torch.sum(spec, axis=1)#.view(16, 512, 256, 2)
                spec = torch.swapaxes(spec, 1, 3)
                spec = torch.swapaxes(spec, 1, 2)

                audio = torch.istft(spec, hop_length=256, center=True, length=65535, n_fft=1022)              #.tolist()#, rate)

                pred_audio = torch.clamp(audio, -1., 1.)


                ground_masks_ = output['ground_masks']
                grid_unwarp = torch.from_numpy(warpgrid(B, stft_frame//2+1, ground_masks_.size(3), warp=False)).to(self.model.device)
                ground_masks_linear = F.grid_sample(ground_masks_, grid_unwarp).view(int(bs/2), 2, 512, 256)

                ground_mag = mag_mix * ground_masks_linear#.cpu().numpy() # [:, 0]

                # spec = ground_mag.astype(np.complex) * np.exp(1j*phase_mix)
                spec = asteroid_filterbanks.transforms.from_magphase(ground_mag, phase_mix, dim = -2).view(16, 2, 2, 512, 256)
                spec = spec[:, 0, :, :, :]
                spec = torch.swapaxes(spec, 1, 3)
                spec = torch.swapaxes(spec, 1, 2)

                
                ground_audio = torch.istft(spec, hop_length=256, center=True, length=65535, n_fft=1022)              #.tolist()#, rate)

                ground_audio = torch.clamp(ground_audio, -1., 1.)
                
                


                coseparation_loss = scale_invariant_signal_distortion_ratio(preds=pred_audio, target=ground_audio)

                    
                lost_loss1 += [coseparation_loss.cpu().detach().numpy()]


                '''
                coseparation_loss = 0

                for idx in range(len(labels)-1, -1, -2):
                    if labels[idx] == 15:
                        #predicted_masks[idx] = predicted_masks[idx] * 0
                        coseparation_loss += self.criterion(predicted_masks[idx - 1], ground_masks[idx])#, weights)
                    else:
                        coseparation_loss += self.criterion(predicted_masks[idx - 1] + predicted_masks[idx], ground_masks[idx])#, weights)
                #coseparation_loss = self.criterion((predicted_masks * vec).view(int(bs / 2), 2, 1, 256, 256), (ground_masks * vec).view(int(bs / 2), 2, 1, 256, 256))#, weights)
                
                coseparation_loss = torch.mean(coseparation_loss)
                '''
                lamda = 0.01
                #consistency_loss = loss.ce_loss(pred_labels, Variable(labels, requires_grad=False)) * lamda
                sum_loss = coseparation_loss    # + consistency_loss


                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', sum_loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(pred_labels, labels))
                self.writer.add_image('input', make_grid(pick['detections'].cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        i = 1
        for name, p in self.model.named_parameters():
            #print(str(i) + ")")
            i += 1
            try:
                #self.writer.add_histogram(name, p, bins='auto')
                pass
            except:
                try:
                    self.writer.add_histogram(name, None, bins='auto')
                except:
                    pass
                print("-->> param dim too large with " + str(name))

        lost_loss += [np.mean(lost_loss1)]
        t += [ind[0]]
        ind[0] += 1
        
        fig, axs = plt.subplots(1, 1)
        axs.set_title("Loss - epoch: " + str(epoch))
        axs.set_ylabel('loss')
        axs.set_xlabel('epochs')
        
        #cop = np.asarray(lost_loss)
        plt.plot(t, lost_loss)
        #plt.label("epoch: " + str(epoch))
        plt.show()
        plt.savefig('val_loss.png')
        np.save('val_loss', lost_loss)

        print("done validate epoch")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)