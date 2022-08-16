from matplotlib.afm import CompositePart
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
from base.base_trainer import BaseTrainer
from utils import inf_loop, MetricTracker
from model import loss
#import torchvision.transforms as T

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

    def _train_epoch(self, epoch):
        print("start train epoch")
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, pick in enumerate(self.data_loader):
            
            self.model.to(self.model.device)
            # num_of_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            # print("Number of parameters that require grad in the model is: {num}".format(num=num_of_param))
            
            #data, target = pick.to(self.device), None#, target.to(self.device)
            #pick['mixed_audio'] = pick['mixed_audio'].view(4, 15, 512, 256)
            #pick['mixed_audio'] = pick['mixed_audio'].to(self.device)
            bs = pick["detections"].shape[0] * pick["detections"].shape[1]
            pick['detections'] = pick['detections'].view(bs, 3, 224, 224)
            #pick['detections'] = pick['detections'].to(self.device)
            #pick['classes'] = pick['classes'].to(self.device)
            
            for key, value in pick.items():
                pick[key] = pick[key].to(self.model.device)

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
            vec = torch.ones(predicted_masks.shape)

            for idx in range(len(labels)-1, -1, -2):  
                 if labels[idx] == 15:
                    vec[idx] = torch.zeros(vec[idx].shape)
                #if labels[idx] != 15:
                    #predicted_masks[idx-1] += predicted_masks[idx]
            
            vec = vec.to(self.model.device)

            coseparation_loss = self.criterion((predicted_masks * vec).view(int(bs / 2), 2, 1, 256, 256), (ground_masks * vec).view(int(bs / 2), 2, 1, 256, 256))#, weights)
            
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
            sum_loss = coseparation_loss #+ consistency_loss
            
            self.optimizer.zero_grad()
            #consistency_loss.backward(retain_graph=True)
            coseparation_loss.backward()

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
            

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        print("done train epoch")
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
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

                for idx in range(len(labels)-1, -1, -2):  
                    if labels[idx] == 15:
                        vec[idx] = torch.zeros(vec[idx].shape)
                    #if labels[idx] != 15:
                        #predicted_masks[idx-1] += predicted_masks[idx]
                
                vec = vec.to(self.model.device)

                coseparation_loss = self.criterion((predicted_masks * vec).view(int(bs / 2), 2, 1, 256, 256), (ground_masks * vec).view(int(bs / 2), 2, 1, 256, 256))#, weights)
                
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