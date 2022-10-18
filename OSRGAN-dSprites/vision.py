import visdom
import torch
from utils import DataGather, make_opt


class viz(object):
    def __init__(self, args):
        # Visdom
        self.viz_on = args.viz_on
        self.win_id = dict(Wasserstein_D='win_Wasserstein_D', 
                           FactorVaeMetric='FactorVaeMetric', SAPMetric='SAPMetric', MIGMetric='MIGMetric')
        self.line_gather = DataGather('iter_loss', 'iter_mertic', 'Wasserstein_D',
                                      'FactorVaeMetric', 'SAPMetric', 'MIGMetric')
        self.code_dim = args.code_dim
        self.name = args.name
        self.compare_line_opt = make_opt(self.code_dim)
        if self.viz_on:
            self.viz_port = args.viz_port
            self.viz = visdom.Visdom()
            self.viz_ll_iter = args.viz_ll_iter
            self.viz_la_iter = args.viz_la_iter
            self.viz_vc_iter = args.viz_vc_iter
            if not self.viz.win_exists(env=self.name + '/lines', win=self.win_id['Wasserstein_D']):
                self.viz_init()

    def visualize_loss_line(self):
        data = self.line_gather.data
        iters = torch.Tensor(data['iter_loss'])
        Wasserstein_D = torch.Tensor(data['Wasserstein_D'])
        self.viz.line(X=iters,
                      Y=Wasserstein_D,
                      env=self.name + '/lines',
                      win=self.win_id['Wasserstein_D'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='Wasserstein Distance', ))

    def visualize_mertic_line(self):
        data = self.line_gather.data
        iters = torch.Tensor(data['iter_mertic'])
        SAPMetric = torch.Tensor(data['SAPMetric'])
        FactorVaeMetric = torch.Tensor(data['FactorVaeMetric'])
        MIGMetric = torch.Tensor(data['MIGMetric'])

        self.viz.line(X=iters,
                      Y=FactorVaeMetric,
                      env=self.name + '/lines',
                      win=self.win_id['FactorVaeMetric'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='FactorVaeMetric', ))
        self.viz.line(X=iters,
                      Y=SAPMetric,
                      env=self.name + '/lines',
                      win=self.win_id['SAPMetric'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='SAPMetric', ))
        self.viz.line(X=iters,
                      Y=MIGMetric,
                      env=self.name + '/lines',
                      win=self.win_id['MIGMetric'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='MIGMetric', ))

    def viz_init(self):
        zero_init = torch.zeros([1])
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name + '/lines',
                      win=self.win_id['Wasserstein_D'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='Wasserstein Distance', ))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name + '/lines',
                      win=self.win_id['FactorVaeMetric'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='FactorVaeMetric', ))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name + '/lines',
                      win=self.win_id['SAPMetric'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='SAPMetric', ))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name + '/lines',
                      win=self.win_id['MIGMetric'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='MIGMetric', ))
