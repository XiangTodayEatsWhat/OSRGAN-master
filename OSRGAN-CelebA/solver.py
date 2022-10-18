import itertools
from operator import length_hint
import imageio
import os
from torch import autograd
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from dataset import return_data
from model import Generator, Discriminator, QHead, DHead, OBEHead
from utils import mkdirs, noise_sample, weights_init_normal, NormalNLLLoss, calculate_activation_statistics, \
    calculate_frechet_distance, create_inception, cov
from vision import viz
from torchvision.utils import make_grid, save_image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


class Solver(object):
    def __init__(self, args):
        use_cuda = args.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
        self.name = args.name
        self.model_type = args.model_type
        self.max_iter = int(args.max_iter)
        self.print_iter = args.print_iter
        self.global_iter = 0
        self.pbar = tqdm(total=self.max_iter)
        self.G_iteration = args.G_iteration
        self.img_size = args.img_size

        # Data
        self.dset_dir = args.dset_dir
        self.batch_size = args.batch_size
        self.nrow = np.int(np.sqrt(self.batch_size))
        self.data_loader = return_data(args)

        # Networks & Optimizers
        self.z_dim = args.z_dim
        self.code_dim = args.code_dim
        self.n_classes = args.n_classes
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.G = Generator().to(self.device)
        self.E = Discriminator().to(self.device)
        self.DHead = DHead().to(self.device)
        self.QHead = QHead().to(self.device)
        self.OBEHead = OBEHead(self.img_size, self.model_type, self.device).to(self.device)
        self.optim_D = torch.optim.Adam(
            itertools.chain(filter(lambda p: p.requires_grad, self.E.parameters()),
                            filter(lambda p: p.requires_grad, self.DHead.parameters())),
            lr=self.lr, betas=(self.beta1, self.beta2)
        )
        self.optim_GQ = torch.optim.Adam(
            itertools.chain(self.G.parameters(), 
                            self.QHead.parameters(),
                            self.OBEHead.parameters()),
            lr=self.lr,
            betas=(self.beta1, self.beta2)
        )
        if len(list(self.OBEHead.parameters())) != 0:
            self.optim_OBE = torch.optim.Adam(self.OBEHead.parameters(), lr=self.lr / 10,
                                              betas=(self.beta1, self.beta2))
        self.nets = [self.G, 
                     self.E, 
                     self.DHead, 
                     self.QHead, 
                     self.OBEHead]

        # loss
        self.I = torch.eye(self.img_size).to(self.device)
        self.mse_loss = torch.nn.MSELoss()
        self.criterionQ_con = NormalNLLLoss()
        self.con_weight = args.con_weight
        self.info_weight = args.info_weight
        self.g_weight = args.g_weight

        # Checkpoint
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
        self.ckpt_save_iter = args.ckpt_save_iter
        mkdirs(self.ckpt_dir)
        self.G.apply(weights_init_normal)
        self.E.apply(weights_init_normal)
        self.DHead.apply(weights_init_normal)
        self.QHead.apply(weights_init_normal)
        if args.ckpt_load:
            self.load_checkpoint(args.ckpt_load)
            self.ckptname = args.ckpt_load

        self.viz = viz(args)
        self.vp_dir = os.path.join(args.vp_dir, args.name)

    def train(self):
        self.net_mode(train=True)
        out = False
        while not out:
            for imgs in self.data_loader:
                self.global_iter += 1
                self.pbar.update(1)

                real_imgs = Variable(imgs.type(self.FloatTensor).to(self.device))
                noise = noise_sample(self.code_dim, self.z_dim, self.batch_size, self.device)
                gen_imgs = self.G(noise)

                # -----------------
                #  train D
                # -----------------
                if self.global_iter % self.G_iteration == 0:
                    self.optim_D.zero_grad()

                    real_pred = self.DHead(self.E(real_imgs))
                    d_real_loss = torch.mean(-real_pred)
                    fake_pred = self.DHead(self.E(gen_imgs.detach()))
                    d_fake_loss = torch.mean(fake_pred)
                    gradient_penalty = self.calc_gradient_penalty(self.E, self.DHead, real_imgs.data)

                    gradient_penalty.backward()
                    d_real_loss.backward()
                    d_fake_loss.backward()

                    d_loss = d_fake_loss.data + d_real_loss.data + gradient_penalty.data

                    self.optim_D.step()

                # -----------------
                #  train Q, G, P
                # -----------------
                self.optim_GQ.zero_grad()

                fake_feature = self.E(gen_imgs)
                validity = self.DHead(fake_feature)
                g_loss = torch.mean(-validity)

                q_mu, q_var = self.QHead(fake_feature)
                info_loss = self.criterionQ_con(noise[:, self.z_dim:], q_mu, q_var)

                coe_c = self.OBEHead(gen_imgs)
                con_loss = self.mse_loss(coe_c[:, -self.code_dim:], noise[:, self.z_dim:])
                
                GQ_loss = self.g_weight * g_loss + self.info_weight * info_loss + self.con_weight * con_loss
                GQ_loss.backward()
                self.optim_GQ.step()

                if self.model_type == 'OBE':
                    # -----------------
                    #  train P
                    # -----------------
                    while True:
                        self.optim_OBE.zero_grad()
                        P = self.OBEHead.infer.P
                        PPI = torch.matmul(P, P.T) - self.I
                        or_loss = torch.sum(torch.abs(PPI))
                        or_loss.backward()
                        self.optim_OBE.step()
                        show_loss = or_loss.cpu().detach().numpy()
                        if show_loss < 0.8:
                            break

                self.net_mode(train=False)
                if self.global_iter % self.print_iter == 0:
                    if self.model_type == 'OBE':
                        self.pbar.write(
                            '[{}] D_loss:{:.3f} G_loss:{:.3f} info_loss:{:.3f} con_loss:{:.3f} or_loss:{:.3f}'.format(
                                self.global_iter, d_loss.item(), g_loss.item(), info_loss.item(),
                                con_loss.item(), show_loss.item()))
                    elif self.model_type == 'DCT':
                        self.pbar.write(
                            '[{}] D_loss:{:.3f} G_loss:{:.3f} info_loss:{:.3f} con_loss:{:.3f}'.format(
                                self.global_iter, d_loss.item(), g_loss.item(), info_loss.item(),
                                con_loss.item()))

                if self.global_iter % self.ckpt_save_iter == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.viz.viz_on and (self.global_iter % self.viz.viz_ll_iter == 0):
                    D_real = real_pred.detach()
                    D_fake = fake_pred.detach()
                    Wasserstein_D = torch.mean(D_real - D_fake)
                    self.viz.line_gather.insert(iter_loss=self.global_iter,
                                                Wasserstein_D=Wasserstein_D.item())

                if self.viz.viz_on and (self.global_iter % self.viz.viz_la_iter == 0):
                    self.viz.visualize_loss_line()
                    self.viz.line_gather.flush()

                if self.viz.viz_on and (self.global_iter % self.viz.viz_vc_iter == 0):
                    self.visualize_z_img()
                    self.visualize_dis_img()

                if self.global_iter >= self.max_iter:
                    out = True
                    break

                self.net_mode(train=True)

        self.pbar.write("[Training Finished]")
        self.pbar.close()

    def calc_gradient_penalty(self, netE, netD, real_data):
        interpolates = real_data.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(netE(interpolates))

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1)) ** 2).mean() * 6
        return gradient_penalty

    def visualize_z_img(self):
        # clarity
        noise = noise_sample(self.code_dim, self.z_dim, self.batch_size, self.device)
        static_sample = self.G(noise)
        title = 'img_fake or real?(iter:{})'.format(self.global_iter)
        self.viz.viz.images(static_sample.data, env=self.name + '/Clarity',
                            opts=dict(title=title), nrow=self.nrow)

    def visualize_dis_img(self):
        # only change one-dimension
        noise = noise_sample(self.code_dim, self.z_dim, 
                             1, self.device).repeat(self.n_classes, 1).repeat(self.code_dim, 1)
        # Variable continuous codes
        varied_c = self.FloatTensor(torch.linspace(-1, 1, self.n_classes).to(self.device))
        for i in range(0, self.code_dim):
            noise[i * self.n_classes:(i + 1) * self.n_classes,
                  self.z_dim + i] = varied_c
        vary_single = self.G(noise)
        title = 'vary_single_c(0-n)(iter:{})'.format(self.global_iter)
        self.viz.viz.images(vary_single.data, env=self.name + '/transpose',
                            opts=dict(title=title), nrow=self.n_classes)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def save_checkpoint(self, ckptname='model', verbose=True):
        model_states = {
            'E': self.E.state_dict(),
            'G': self.G.state_dict(),
            'DHead': self.DHead.state_dict(),
            'QHead': self.QHead.state_dict(),
            'OBEHead': self.OBEHead.state_dict(),
        }

        optim_states = {
            'optim_D': self.optim_D.state_dict(),
            'optim_GQ': self.optim_GQ.state_dict(),
            'optim_OBE': self.optim_OBE.state_dict()
        }
        states = {
            'iter': self.global_iter,
            'model_states': model_states,
            'optim_states': optim_states
        }

        filepath = os.path.join(self.ckpt_dir, ckptname)
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        if verbose:
            self.pbar.write("=> saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))

    def load_checkpoint(self, ckptname='last', verbose=True):
        if ckptname == 'last':
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                if verbose:
                    self.pbar.write("=> no checkpoint found")
                return

            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])

        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']
            self.G.load_state_dict(checkpoint['model_states']['G'])
            self.E.load_state_dict(checkpoint['model_states']['E'])
            self.DHead.load_state_dict(checkpoint['model_states']['DHead'])
            self.QHead.load_state_dict(checkpoint['model_states']['QHead'])
            self.OBEHead.load_state_dict(checkpoint['model_states']['OBEHead'])
            self.optim_GQ.load_state_dict(checkpoint['optim_states']['optim_GQ'])
            self.optim_D.load_state_dict(checkpoint['optim_states']['optim_D'])
            self.optim_OBE.load_state_dict(checkpoint['optim_states']['optim_OBE'])
            self.pbar.update(self.global_iter)
            if verbose:
                self.pbar.write("=> loaded checkpoint '{}'".format(filepath))
        else:
            if verbose:
                self.pbar.write("=> no checkpoint found at '{}'".format(filepath))

    # Clarity metric
    def eval_single_model(self):
        self.net_mode(train=False)
        with torch.no_grad():
            self.gen_vp()
            fid = self.gen_fid()
            self.pbar.write("fid metric:{}".format(fid))
        
        
    def gen_fid(self, fid_N=128000):
        # make data
        inception = create_inception()
        fake_inception_activations = torch.ones(fid_N, 2048).type(
            self.FloatTensor).to(self.device)
        real_inception_activations = torch.ones(fid_N, 2048).type(
            self.FloatTensor).to(self.device)
        step = 0
        fid_bar = tqdm(total=fid_N // self.batch_size)
        for imgs in self.data_loader:
            if self.batch_size * step >= fid_N:
                break
            fid_bar.update(1)
            # ready to work
            noise = noise_sample(self.code_dim, self.z_dim, self.batch_size,
                                 self.device)
            fake = self.G(noise)
            real_imgs = Variable(imgs.type(self.FloatTensor).to(self.device))
            fake_activations_batch = calculate_activation_statistics(
                fake, inception)
            real_activations_batch = calculate_activation_statistics(
                real_imgs, inception)
            frm = step * self.batch_size
            to = frm + self.batch_size
            fake_inception_activations[frm:to, :] = fake_activations_batch
            real_inception_activations[frm:to, :] = real_activations_batch
            step += 1
        m1 = torch.mean(fake_inception_activations, dim=0).cpu()
        s1 = cov(fake_inception_activations, rowvar=False).cpu()
        m2 = torch.mean(real_inception_activations, dim=0).cpu()
        s2 = cov(real_inception_activations, rowvar=False).cpu()
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

    def gen_vp(self, N=15000):
        mkdirs(self.vp_dir)
        label = np.zeros((N, self.code_dim))
        n = int(N / self.code_dim)
        num = 0
        for sample_type in tqdm(range(n)):
            noise = noise_sample(self.code_dim, self.z_dim, 1, self.device)
            noise1 = noise.repeat(self.code_dim, 1)
            noise2 = noise.repeat(self.code_dim, 1)
            for i in range(self.code_dim):
                varied_c = self.FloatTensor((torch.rand(1) * 2 - 1).to(self.device))
                noise2[i, self.z_dim + i] = varied_c
                label[i + sample_type * self.code_dim, i] = 1
            x1 = self.G(noise1)
            x2 = self.G(noise2)
            x = torch.cat([x1, x2], dim=-1).permute(0, 2, 3, 1).cpu().detach().numpy() * 255
            for img in x:
                imageio.imwrite(os.path.join(self.vp_dir, 'pair_{:06d}.jpg'.format(num)), img.astype('uint8'))
                num += 1
            print(sample_type)
        np.save(os.path.join(self.vp_dir, 'labels.npy'), label)    