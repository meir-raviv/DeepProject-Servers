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
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, pick in enumerate(self.data_loader):
            
            #data, target = pick.to(self.device), None#, target.to(self.device)
            #pick['mixed_audio'] = pick['mixed_audio'].view(4, 15, 512, 256)
            pick['mixed_audio'] = pick['mixed_audio'].to(self.device)
            bs = pick["detections"].shape[0] * pick["detections"].shape[1]
            pick['detections'] = pick['detections'].view(bs, 3, 224, 224)
            pick['detections'] = pick['detections'].to(self.device)
            pick['classes'] = pick['classes'].to(self.device)
            

            self.optimizer.zero_grad()
            output = self.model(pick)

            ground_masks = output["ground_masks"].numpy()
            #print(ground_masks.shape)
            #ground_masks = ground_masks.view(32, ground_masks.shape[2], ground_masks.shape[3])
            predicted_spectrograms = output["predicted_masks"].clone().detach().numpy()
            #print(predicted_spectrograms.shape)
            #predicted_spectrograms = predicted_spectrograms.view(32, predicted_spectrograms.shape[2], predicted_spectrograms.shape[3])
            weights = output["weights"]
            #print(weights.shape)
            weights = weights.view(bs, weights.shape[2], weights.shape[3]).numpy()

            #start from 0?
            labels = output["ground_labels"].view(-1).numpy()   #.astype(np.float32)
            
            pred_labels= output['predicted_audio_labels'].clone().detach().numpy()  #.view(-1).numpy()
            #pred_labels = np.zeros(32)
            
            #for idx in range(len(predicted_labels)-1, -1, -1):
            #   pred_labels[idx] = np.argmax(predicted_labels[idx])
            
            #print(ground_masks[0])

            for idx in range(len(labels)-1, -1, -1):
                #print(labels[idx])
                if labels[idx] == -2:
                    #print(idx)
                    ground_masks = np.delete(ground_masks, idx, axis=0)
                    predicted_spectrograms = np.delete(predicted_spectrograms, idx, axis=0)
                    weights = np.delete(weights, idx, axis=0)
                    pred_labels = np.delete(pred_labels, idx, axis=0)
                    labels = np.delete(labels, idx, axis=0)
            
            #print("-------")
            #print(pred_labels.shape)
            
            for idx in range(len(predicted_spectrograms)):
#                predicted_spectrograms[idx] = torch.clamp(torch.from_numpy(predicted_spectrograms[idx]), 0, 1)
                predicted_spectrograms[idx] = np.clip(predicted_spectrograms[idx], 0, 1)
                ground_masks[idx] = np.clip(ground_masks[idx], 0, 1)

                #should we clamp input also>?????
            
            weights = np.expand_dims(weights, axis=1)
            ground_masks = torch.from_numpy(ground_masks)

            #??????????????
            predicted_spectrograms = predicted_spectrograms[:, 0, :, :]
            predicted_spectrograms = np.expand_dims(predicted_spectrograms, axis = 1)
            predicted_spectrograms = torch.from_numpy(predicted_spectrograms)
            
            weights = torch.from_numpy(weights)

            pred_labels = torch.from_numpy(pred_labels)
            labels = torch.from_numpy(labels)

            #print(ground_masks.shape)
            #print(predicted_spectrograms.shape)
            #print(weights.shape)

            # should we mul with weights the result of loss?
            coseparation_loss = self.criterion(Variable(predicted_spectrograms, requires_grad=True), Variable(ground_masks, requires_grad=False), weights)
            
            #print("-------")
            #print(pred_labels.shape)
            #print(labels.shape)

            consistency_loss = loss.ce_loss(Variable(pred_labels, requires_grad=True), Variable(labels, requires_grad=False)) * 0.01    #lambda
            sum_loss = consistency_loss + coseparation_loss
            consistency_loss.backward(retain_graph=True)
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
                pick['mixed_audio'] = pick['mixed_audio'].to(self.device)
                bs = pick["detections"].shape[0] * pick["detections"].shape[1]
                pick['detections'] = pick['detections'].view(bs, 3, 224, 224)
                pick['detections'] = pick['detections'].to(self.device)
                pick['classes'] = pick['classes'].to(self.device)
                
                output = self.model(pick)

                ground_masks = output["ground_masks"].numpy()
                predicted_spectrograms = output["predicted_masks"].clone().detach().numpy()
                weights = output["weights"]
                weights = weights.view(bs, weights.shape[2], weights.shape[3]).numpy()

                labels = output["ground_labels"].view(-1).numpy()   #.astype(np.float32)
                
                pred_labels= output['predicted_audio_labels'].clone().detach().numpy()  #.view(-1).numpy()
                

                for idx in range(len(labels)-1, -1, -1):
                    if labels[idx] == -2:
                        ground_masks = np.delete(ground_masks, idx, axis=0)
                        predicted_spectrograms = np.delete(predicted_spectrograms, idx, axis=0)
                        weights = np.delete(weights, idx, axis=0)
                        pred_labels = np.delete(pred_labels, idx, axis=0)
                        labels = np.delete(labels, idx, axis=0)
                
                for idx in range(len(predicted_spectrograms)):
                    predicted_spectrograms[idx] = np.clip(predicted_spectrograms[idx], 0, 1)
                    ground_masks[idx] = np.clip(ground_masks[idx], 0, 1)
                
                weights = np.expand_dims(weights, axis=1)
                ground_masks = torch.from_numpy(ground_masks)

                predicted_spectrograms = predicted_spectrograms[:, 0, :, :]
                predicted_spectrograms = np.expand_dims(predicted_spectrograms, axis = 1)
                predicted_spectrograms = torch.from_numpy(predicted_spectrograms)
                
                weights = torch.from_numpy(weights)

                pred_labels = torch.from_numpy(pred_labels)
                labels = torch.from_numpy(labels)

                coseparation_loss = self.criterion(Variable(predicted_spectrograms, requires_grad=True), Variable(ground_masks, requires_grad=False), weights)
                

                consistency_loss = loss.ce_loss(Variable(pred_labels, requires_grad=True), Variable(labels, requires_grad=False)) * 0.01    #lambda
                sum_loss = consistency_loss + coseparation_loss


                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', sum_loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(pred_labels, labels))
                self.writer.add_image('input', make_grid(pick['detections'].cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
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