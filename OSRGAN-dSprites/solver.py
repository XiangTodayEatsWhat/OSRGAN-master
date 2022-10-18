import itertools
import os
import math
import numpy as np
import torch
from torch import autograd
from torch.autograd import Variable
from tqdm import tqdm

from dataset import return_data
from model import Generator, Discriminator, QHead, DHead, OBEHead
from utils import mkdirs, noise_sample, weights_init_normal, NormalNLLLoss
from sklearn.svm import LinearSVC
from sklearn.metrics import mutual_info_score
from vision import viz



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
        self.metric_times = args.metric_times
        self.img_size = args.img_size

        # Data
        self.dset_dir = args.dset_dir
        self.batch_size = args.batch_size
        self.nrow = np.int(np.sqrt(self.batch_size))
        self.imgs, self.data_loader, self.latent_values, self.metadata = return_data(args)
        self.latents_sizes = self.metadata["latents_sizes"]
        self.latents_bases = np.concatenate(
            (self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

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
            
        self.viz = viz(args)
        self.seed = args.seed
        
        # outputs
        self.metric_log_dir = args.metric_log_dir
        mkdirs(self.metric_log_dir)
        self.metric_log = open(os.path.join(self.metric_log_dir, args.name + '.txt'), 'w')


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
                        if show_loss < 0.2:
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
                    self.Metric()
                    self.save_checkpoint(str(self.global_iter))
                    
                if self.viz.viz_on and (self.global_iter % self.viz.viz_ll_iter == 0):
                    D_real = real_pred.detach()
                    D_fake = fake_pred.detach()
                    Wasserstein_D = torch.mean(D_real - D_fake)
                    self.viz.line_gather.insert(iter_loss=self.global_iter,
                                                Wasserstein_D=Wasserstein_D.item())

                if self.viz.viz_on and (self.global_iter % self.viz.viz_la_iter == 0):
                    self.visualize_coefficient_line()    
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

    def Metric(self):
        factorVAE_metric = self.FactorVAEmetric()
        SAP_metric = self.SAPmetric()
        MIG_metric = self.MIGmetric()
        self.viz.line_gather.insert(iter_mertic=self.global_iter,
                                    FactorVaeMetric=factorVAE_metric,
                                    SAPMetric=SAP_metric,
                                    MIGMetric=MIG_metric,
                                    )
        self.viz.visualize_mertic_line()
        self.viz.line_gather.flush()
        self.metric_log.write('FV:' + str(factorVAE_metric) + ' SAP:' + str(SAP_metric) + ' MIG:' + str(
            MIG_metric) + ' iters:' + str(self.global_iter) + '\n')
        self.metric_log.flush()

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
        return samples

    def sample_img_with_latents(self):
        random_latent_ids = self.sample_latent(size=int(self.imgs.shape[0] / 10))
        random_latent_ids = random_latent_ids.astype(np.int32)
        random_ids = self.latent_to_index(random_latent_ids)
        assert random_latent_ids.shape == (int(self.imgs.shape[0] / 10), 6)
        random_imgs = self.imgs[random_ids]
        random_latents = self.latent_values[random_ids]

        return random_imgs, random_latents, random_latent_ids

    def FactorVAEmetric(self):
        # Calculate the empirical standard deviation
        selected_ids = np.random.permutation(range(self.imgs.shape[0]))
        selected_ids = selected_ids[0: int(self.imgs.shape[0] / 10)]
        metric_data_eval_std = self.imgs[selected_ids]

        eval_std_inference = self.inference_from(metric_data_eval_std)
        eval_std = np.std(eval_std_inference, axis=0, keepdims=True)

        # batchsize 128 iteration 500
        L = 100
        M = 500
        train_data = np.zeros((eval_std.shape[1], 5))

        for i in range(M):
            fixed_latent_id = i % 5 + 1
            latents_sampled = self.sample_latent(size=L)
            latents_sampled[:, fixed_latent_id] = \
                np.random.randint(self.latents_sizes[fixed_latent_id], size=1)
            indices_sampled = self.latent_to_index(latents_sampled)
            imgs_sampled = self.imgs[indices_sampled]

            data_inference = self.inference_from(imgs_sampled)
            data_inference /= eval_std
            data_std = np.std(data_inference, axis=0)
            predict = np.argmin(data_std)
            label = fixed_latent_id - 1
            train_data[predict, label] += 1

        total_sample = np.sum(train_data)
        maxs = np.amax(train_data, axis=1)
        correct_sample = np.sum(maxs)
        factorVAE_metric = float(correct_sample) / total_sample
        return factorVAE_metric

    def SAPmetric(self):
        random_imgs, random_latents, _ = self.sample_img_with_latents()

        data_inference = self.inference_from(random_imgs)
        data_gt_latents = random_latents
        factor_is_continuous = [False, True, True, True, True]

        num_latents = data_inference.shape[1]
        num_factors = len(factor_is_continuous)

        score_matrix = np.zeros([num_latents, num_factors])
        for i in range(num_latents):
            for j in range(num_factors):
                inference_values = data_inference[:, i]
                gt_values = data_gt_latents[:, j]
                if factor_is_continuous[j]:
                    cov = np.cov(inference_values, gt_values, ddof=1)
                    assert np.all(np.asarray(list(cov.shape)) == 2)
                    cov_cov = cov[0, 1] ** 2
                    cov_sigmas_1 = cov[0, 0]
                    cov_sigmas_2 = cov[1, 1]
                    score_matrix[i, j] = cov_cov / cov_sigmas_1 / cov_sigmas_2
                else:
                    gt_values = gt_values.astype(np.int32)
                    classifier = LinearSVC(C=0.01, class_weight="balanced")
                    classifier.fit(inference_values[:, np.newaxis], gt_values)
                    pred = classifier.predict(inference_values[:, np.newaxis])
                    score_matrix[i, j] = np.mean(pred == gt_values)

        sorted_score_matrix = np.sort(score_matrix, axis=0)
        score = np.mean(sorted_score_matrix[-1, :] -
                        sorted_score_matrix[-2, :])

        return score

    def MIGmetric(self):
        def discretize(data, num_bins=20):
            discretized = np.zeros_like(data)
            for i in range(data.shape[1]):
                discretized[:, i] = np.digitize(
                    data[:, i],
                    np.histogram(data[:, i], num_bins)[1][:-1])
            return discretized

        def mutual_info(data1, data2):
            n1 = data1.shape[1]
            n2 = data2.shape[1]
            mi = np.zeros([n1, n2])
            for i in range(n1):
                for j in range(n2):
                    mi[i, j] = mutual_info_score(
                        data2[:, j], data1[:, i])
            return mi

        def entropy(data):
            num_factors = data.shape[1]
            entr = np.zeros(num_factors)
            for i in range(num_factors):
                entr[i] = mutual_info_score(data[:, i], data[:, i])
            return entr

        random_imgs, _, random_latent_ids = self.sample_img_with_latents()

        data_inference = self.inference_from(random_imgs)
        data_gt_latents = random_latent_ids[:, 1:]

        data_inference_discrete = discretize(data_inference)
        mi = mutual_info(
            data_inference_discrete, data_gt_latents)
        etro = entropy(data_gt_latents)
        sorted_mi = np.sort(mi, axis=0)[::-1]
        mig_score = np.mean(
            np.divide(sorted_mi[0, :] - sorted_mi[1, :], etro))

        return mig_score

    def inference_from(self, img):
        latents = []
        img = torch.from_numpy(img).type(self.FloatTensor).to(device=self.device)
        for i in range(int(math.ceil(float(img.shape[0]) / self.batch_size))):
            feature = self.E(img[i * self.batch_size:(i + 1) * self.batch_size])
            sub_latents, _ = self.QHead(feature)
            sub_latents = sub_latents.cpu().detach().numpy()
            latents.append(sub_latents)
        return np.vstack(latents)

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
        

    def visualize_coefficient_line(self, sample_num=32):
        # Observe whether the coefficient on the corresponding basis of the generated picture and c change similarly
        noise = noise_sample(self.code_dim, self.z_dim, 1, self.device).repeat(sample_num, 1).repeat(self.code_dim, 1)
        # Variable continuous codes
        varied_c = self.FloatTensor(torch.linspace(-1, 1, sample_num).to(self.device))
        for i in range(self.code_dim):
            noise[i * sample_num:(i + 1) * sample_num, self.z_dim + i] = varied_c
        coe_c = self.OBEHead(self.G(noise))
        for i in range(self.code_dim):
            Y = coe_c[i * sample_num:(i + 1) * sample_num, -self.code_dim:].cpu().detach().numpy().tolist()
            self.viz.viz.line(X=varied_c.data, Y=Y, env=self.name + '/compare_line', win='chart{}'.format(i),
                              opts=self.viz.compare_line_opt[i])

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def save_checkpoint(self, ckptname='model', verbose=True):
        model_states = {'E': self.E.state_dict(),
                        'G': self.G.state_dict(),
                        'DHead': self.DHead.state_dict(),
                        'QHead': self.QHead.state_dict(),
                        'OBEHead': self.OBEHead.state_dict(),
                        }

        optim_states = {'optim_D': self.optim_D.state_dict(),
                        'optim_GQ': self.optim_GQ.state_dict(),
                        'optim_OBE': self.optim_OBE.state_dict()
                        }
        
        states = {'iter': self.global_iter,
                  'model_states': model_states,
                  'optim_states': optim_states}

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

    def eval_single_model(self):
        self.net_mode(train=False)
        factorVAE_metric = self.FactorVAEmetric()
        SAP_metric = self.SAPmetric()
        MIG_metric = self.MIGmetric()
        self.pbar.write("factorVAE_metric:{}, SAP_metric:{}, MIG_metric:{}"
                        .format(factorVAE_metric,
                                SAP_metric,
                                MIG_metric))