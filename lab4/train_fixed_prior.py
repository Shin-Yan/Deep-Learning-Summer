import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, plot_pred, plot_rec, finn_eval_seq, pred

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=100, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=True, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_monotonic', default=False, action='store_true', help='use monotonic mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=2, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=11, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')  

    args = parser.parse_args()
    return args

def train(x, cond, modules, optimizer, kl_anneal, args, device):
    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()

    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    mse = 0
    kld = 0
    mse_criterion = nn.MSELoss()
    x = x.to(device)
    cond = cond.to(device)
    use_teacher_forcing = True if random.random() < args.tfr else False
    for i in range(1, args.n_past + args.n_future):
        # Take ground truth as input for training
        if use_teacher_forcing is True:
            h = modules['encoder'](x[i-1].type(torch.cuda.FloatTensor))
            h_t = (modules['encoder'](x[i].type(torch.cuda.FloatTensor)))[0]
            if args.last_frame_skip or i < args.n_past:
                h, skip = h
            else:
                h = h[0]
            z , mu , logvar = modules['posterior'](h_t)
            h = h.to(device)
            z = z.to(device)

            h_pred = modules['frame_predictor'](torch.cat([h,z],1), cond[i])
            x_pred = modules['decoder']([h_pred, skip])
        # Doesn't use teacher forcing
        else:
            if i > 1:
                h = modules['encoder'](temp.type(torch.cuda.FloatTensor))
            else:
                h = modules['encoder'](x[i-1].type(torch.cuda.FloatTensor))
            h_t = (modules['encoder'](x[i].type(torch.cuda.FloatTensor)))[0]
            if args.last_frame_skip or i < args.n_past:
                h, skip = h
            else:
                h = h[0]
            z, mu, logvar = modules['posterior'](h_t)
            h = h.to(device)
            z = z.to(device)
            tempinput = torch.cat([h,z],1)
            h_pred = modules['frame_predictor'](tempinput, cond[i])
            x_pred = modules['decoder']([h_pred, skip])
            temp = x_pred
        mse += mse_criterion(x_pred, x[i])
        kld += kl_criterion(mu, logvar, args)

    beta = kl_anneal.get_beta()
    loss = mse + kld * beta
    loss.backward()

    optimizer.step()

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past)

def Unsqueeze(args, seq):
    for i in range(args.batch_size):
        for j in range(args.batch_size):
            if(j==0):
                temp_tensor = torch.unsqueeze(seq[j][i], dim = 0)
            else:
                temp_tensor = torch.cat((temp_tensor, torch.unsqueeze(seq[j][i], dim=0)), dim=0)
        if(i==0):
            temp_tensor_out = torch.unsqueeze(temp_tensor, dim = 0)
        else:
            temp_tensor_out = torch.cat((temp_tensor_out, torch.unsqueeze(temp_tensor, dim=0)), dim=0)
    
    return temp_tensor_out

class kl_annealing():
    def __init__(self, args):
        super().__init__()
        self.cyclical = args.kl_anneal_cyclical
        self.monotonic = args.kl_anneal_monotonic
        self.kl_anneal_ratio = args.kl_anneal_ratio
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.beta = args.beta
        self.niter = args.niter
        self.epoch = 0
        self.epoch_size = args.epoch_size
        self.period = self.niter / args.kl_anneal_cycle
        self.step = (1.0) * self.kl_anneal_ratio / self.period

    def update(self):
        if self.monotonic:
            if self.epoch < 40:
                self.beta = 0.0
            self.beta = 0.002 * (self.epoch - 40)
            if self.beta >= 1.0:
                self.beta = 1.0
        else:
            self.monotonic = False
            if self.epoch % self.period == 0:
                self.beta = 0.0001
            self.beta += self.step
            if self.beta >= 1.0:
                self.beta = 1.0
            self.epoch += 1
        return
    
    def get_beta(self):
        return self.beta

def teacher_forcing_ratio(epoch):
    if epoch < 150:
        return 1.0
    ratio = 1.0 -0.005 * (epoch -150)
    if ratio < 0.0:
        return ratio
    return ratio

def main():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        torch.cuda.set_device(0)
        device = 'cuda'
    else:
        device = 'cpu'
    
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    else:
        name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f'\
            % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)

        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))
    
    print(args)

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # ------------ build the models  --------------

    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
    else:
        frame_predictor = lstm(args.g_dim+args.z_dim, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
            
    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    validate_data = bair_robot_pushing_dataset(args, 'validate')
    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    train_iterator = iter(train_loader)

    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    validate_iterator = iter(validate_loader)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    kl_anneal = kl_annealing(args)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }
    # --------- training loop ------------------------------------

    progress = tqdm(total=args.niter)
    best_val_psnr = 0
    plot_losses = {'epoch' : [], 'KLD': [], 'CrossEntropy': [], 'KLD Weight': [], 'Teacher Forcing Ratio': []}
    for epoch in range(start_epoch, start_epoch + niter):
        frame_predictor.train()
        posterior.train()
        encoder.train()
        decoder.train()

        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0
        kl_anneal.update()

        for _ in range(args.epoch_size):
            try:
                seq, cond = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                seq, cond = next(train_iterator)
            
            # Sequence
            tempseq_tensor_out = Unsqueeze(args, seq)
    
            # Condition
            tempcond_tensor_out = Unsqueeze(args, cond)

            loss, mse, kld = train(seq, cond, modules, optimizer, kl_anneal, args, device)
            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld
        # Save for plotting
        plot_losses['epoch'].append(epoch)
        plot_losses['KLD'].append(epoch_kld)
        plot_losses['CrossEntropy'].append(epoch_loss)
        plot_losses['KLD Weight'].append(kl_anneal.get_beta())
        plot_losses['Teacher Forcing Ratio'].append(args.tfr)
        if epoch >= args.tfr_start_decay_epoch:
            slope = 0.01
            times = 10
            args.tfr = 1.0 - (slope*(epoch//times))
            if args.tfr <= 0.0:
                args.tfr = 0.0

        progress.update(1)
        with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f\n' % (epoch, epoch_loss  / args.epoch_size, epoch_mse / args.epoch_size, epoch_kld / args.epoch_size)))
        
        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()

        if epoch % 5 == 0:
            psnr_list = []
            for _ in range(len(validate_data) // args.batch_size):
                try:
                    validate_seq, validate_cond = next(validate_iterator)
                except StopIteration:
                    validate_iterator = iter(validate_loader)
                    validate_seq, validate_cond = next(validate_iterator)
                # Sequence
                tempseq_tensor_out = Unsqueeze(args, validate_seq)
               
                # Condition
                tempcond_tensor_out = Unsqueeze(args, validate_cond)

                pred_seq = pred(tempseq_tensor_out, tempcond_tensor_out, modules, args, device)
                _, _, psnr = finn_eval_seq(tempseq_tensor_out[args.n_past:], pred_seq[args.n_past:])
                psnr_list.append(psnr)
                
            ave_psnr = np.mean(np.concatenate(psnr))
            print(f'AVG PSNR: {ave_psnr}')

            with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write(('====================== validate psnr = {:.5f} ========================\n'.format(ave_psnr)))

            if ave_psnr > best_val_psnr:
                best_val_psnr = ave_psnr
                # save the model
                torch.save({
                    'encoder': encoder,
                    'decoder': decoder,
                    'frame_predictor': frame_predictor,
                    'posterior': posterior,
                    'args': args,
                    'last_epoch': epoch},
                    '%s/model.pth' % args.log_dir)
            print(f'Best Validate PSNR: {best_val_psnr}')

        if epoch % 20 == 0:
            try:
                validate_seq, validate_cond = next(validate_iterator)
            except StopIteration:
                validate_iterator = iter(validate_loader)
                validate_seq, validate_cond = next(validate_iterator)

            # Sequence
            tempseq_tensor_out = Unsqueeze(args, validate_seq)
            
            # Condition
            tempcond_tensor_out = Unsqueeze(args, validate_cond)

            plot_pred(validate_seq, validate_cond, modules, epoch, args, device)
            plot_rec(validate_seq, validate_cond, modules, epoch, args, device, plot_losses)

if __name__ == '__main__':
    main()
        
