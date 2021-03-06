# Using a MLP structure and Wasserstein Metric

import time
import sys
import numpy as np
from torch import nn
from torch import optim
import torch
import torch.nn.functional as F
from data_reader import get_trajectory
from config import set_trim_length
import config
from data_analyzer import average_cca_score
from data_analyzer import average_spectra_diff_score
from data_analyzer import average_spectra_cca_score
from data_analyzer import get_single_side_frequency
from data_processor import fft_all_data
from data_processor import fft_data
from data_processor import ifft_data
from data_processor import iflatten_complex_data_with
from data_processor import low_pass_filter_list
from data_processor import lpf_dimension_reduction
from data_processor import window
from data_processor import iwindow
from data_processor import data_center_part
from data_processor import pad_data_zeros
from data_processor import pca_data
from data_processor import standardize_all_data
from data_processor import flatten_complex_data
from data_processor import iflatten_complex_data
from data_processor import time_stamp
from data_processor import trim_data
from dynamic_reporter import init_dynamic_report
from dynamic_reporter import stop_dynamic_report
from dynamic_reporter import report
from data_reader import write_one_file
from multiprocessing import set_start_method
import random
import os

# Prepare the training set for this model
print('Preparing the training set...')
if config.TRIM_LENGTH is None:
    set_trim_length(300)
train_set = trim_data(standardize_all_data())
# train_set, WIN = window(train_set, 'hamming')
train_set = fft_data(train_set)
train_set, dim = lpf_dimension_reduction(train_set, frequency=10)
train_set = flatten_complex_data(train_set)
print(dim)
print('Training set is ready!')

class Complex_Fully_Connected_Linear_Discriminator_LPF(nn.Module):
    def __init__(self, dimension):
        super(Complex_Fully_Connected_Linear_Discriminator_LPF, self).__init__()
        self.n_in = dimension * 2 * 6   # real part and imaginary part are saperated

        # hidden linear layers
        self.linear1 = nn.Linear(self.n_in, self.n_in * 8)
        self.linear2 = nn.Linear(self.n_in * 8, self.n_in * 8)
        self.linear3 = nn.Linear(self.n_in * 8, self.n_in * 8)
        self.linear4 = nn.Linear(self.n_in * 8, 1)

        return

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), 0.1)
        x = F.leaky_relu(self.linear2(x), 0.1)
        x = F.leaky_relu(self.linear3(x), 0.1)
        x = self.linear4(x)

        return x


class Complex_Fully_Connected_Generator_LPF(nn.Module):
    def __init__(self, dimension):
        super(Complex_Fully_Connected_Generator_LPF, self).__init__()
        self.n_in = 10
        self.n_out = dimension * 2 * 6

        # linear layers
        self.linear1 = nn.Linear(self.n_in, self.n_out * 8)
        self.linear2 = nn.Linear(self.n_out * 8, self.n_out * 8)
        self.linear3 = nn.Linear(self.n_out * 8, self.n_out * 8)
        self.linear4 = nn.Linear(self.n_out * 8, self.n_out)

        return

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), 0.1)
        x = F.leaky_relu(self.linear2(x), 0.1)
        x = F.leaky_relu(self.linear3(x), 0.1)
        out = self.linear4(x)

        return out


class Complex_Fully_Connected_WGAN_LPF(nn.Module):
    def __init__(self, dimension):
        super(Complex_Fully_Connected_WGAN_LPF, self).__init__()
        self.dimension = dimension
        self.generator = Complex_Fully_Connected_Generator_LPF(dimension)
        self.discriminator = Complex_Fully_Connected_Linear_Discriminator_LPF(dimension)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_cpu = True

        # Variables to record training process
        self.losses = []
        self.scores = []

        return

    def forward(self, x):
        return self.generator.forward(x)

    def generate(self, n):
        return self.forward(self.noise(n))

    def example(self):
        # Generate an example with numpy array data type
        x = self.generate(1).detach().cpu()
        x = np.array(x)
        x = iflatten_complex_data_with(x, 16)
        x = pad_data_zeros(x, get_single_side_frequency().shape[0])
        x = ifft_data(x)
        # x = iwindow(x, WIN, 0.8)

        return x[0]

    def noise(self, batch_size):
        if self.in_cpu:
            return torch.randn([batch_size, 10])

        return torch.randn([batch_size, 10], device=self.device)

    def train(self, train_set, batch_size, num_epoche, g_eta, d_eta, n_critic, clip_value, show=True):
        print('Start training | batch_size:{a} | eta:{b}'.format(a=batch_size, b=g_eta))
        global origin
        self.to(self.device)
        self.in_cpu = False
        train_set = torch.tensor(train_set, dtype=torch.float, device=self.device)

        g_optimizer = optim.RMSprop(self.generator.parameters(), lr=g_eta)
        d_optimizer = optim.RMSprop(self.discriminator.parameters(), lr=d_eta)

        # g_target = torch.ones(batch_size, 1).to(self.device)
        # d_target = torch.cat([torch.zeros(batch_size, 1), torch.ones(batch_size, 1)], 0).to(self.device)

        N = train_set.size()[0]
        N = N - N % batch_size

        best_ws_dist = sys.float_info.max

        for epoch in range(num_epoche):
            tic = time.time()

            perm = torch.randperm(N)
            
            steps = 0
            for i in range(0, N, batch_size):
                # optimize discriminator
                d_optimizer.zero_grad()
                indices = perm[i:i + batch_size]

                fake = self.generate(batch_size).detach()
                real = train_set[indices]
                
                # The critic loss, also the negative Wassertein distance estimate
                # The discriminator/critic tries to maxmize the wassertein distance
                d_loss = -torch.mean(self.discriminator(real)) + torch.mean(self.discriminator(fake))

                d_loss.backward()
                d_optimizer.step()

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

                # optimize generator every n_critic iterations
                if i % n_critic == 0:
                    g_optimizer.zero_grad()

                    fake = self.generate(batch_size)

                    g_loss = -torch.mean(self.discriminator(fake))

                    g_loss.backward()
                    g_optimizer.step()

                steps += 1
                if show:
                    if steps % 100 == 0:
                        # record training losses
                        g_r = float(g_loss.detach().cpu())
                        d_r = float(d_loss.detach().cpu())
                        self.losses.append([g_r, d_r])

                        # record model score
                        ws_dist = -d_r

                        self.scores.append([ws_dist])

                        if ws_dist < best_ws_dist:
                            best_ws_dist = ws_dist
                            if epoch > 0:
                                torch.save(self.state_dict(), 'BEST_WS')

                        report(
                            loss_title='Training loss curve',
                            losses=self.losses,
                            loss_labels=['Generator', 'Discriminator'],
                            score_title='Model score curve',
                            scores=self.scores,
                            score_labels=['Wasserstein estimate'],
                            interval=100,
                            example=self.example()
                        )

            dt = time.time() - tic
            print('epoch ' + str(epoch) + 'finished! Time usage: ' + str(dt))

            # if show is True:
            #     with torch.no_grad():
            #         y = self.generate(1).to(torch.device('cpu'))
            #     y = iflatten_complex_data(y)
            #     diff = average_spectra_diff(y[0])
            #     print('The Spectra Difference: ' + str(diff))
        
        self.to(torch.device('cpu'))
        self.in_cpu = True

        last_path = os.path.join(
            config.DATA_PATH,
            'Trained_Models',
            'Complex_Fully_Connected_WGAN_LPF',
            time_stamp() + '|LAST' + '|BC:' + str(batch_size) + '|g_eta:' + str(g_eta) + '|d_eta:' + str(d_eta) + '|n_critic:' + str(n_critic) + '|clip_value:' + str(clip_value)
        )
        torch.save(self.state_dict(), last_path)

        # Store the model with lest Wasserstein estimate
        try:
            self.load_state_dict(torch.load('BEST_WS'))
            os.remove('BEST_WS')

            best_asd_path = os.path.join(
                config.DATA_PATH,
                'Trained_Models',
                'Complex_Fully_Connected_WGAN_LPF',
                time_stamp() + '|WS' + '|BC:' + str(batch_size) + '|g_eta:' + str(g_eta) + '|d_eta:' + str(d_eta) + '|n_critic:' + str(n_critic) + '|clip_value:' + str(clip_value)
            )
            torch.save(self.state_dict(), best_asd_path)
        except:
            pass

        return


if __name__ == '__main__':
    set_start_method('spawn')   # To make dynamic reporter works

    for eta in [0.00001, 0.0001, 0.001]:
        for i in range(3):
            report_path = os.path.join(
                config.DATA_PATH,
                'Training_Reports',
                'Complex_Fully_Connected_WGAN_LPF',
                time_stamp() + '|eta:' + str(eta) + '|n_critic:' + str(5) + '|clip_value:' + str(0.01) + '.png'
            )
            init_dynamic_report(5, report_path)
            gan = Complex_Fully_Connected_WGAN_LPF(dim)
            gan.train(train_set, 10, 300, eta, eta, 5, 0.01, True)
            stop_dynamic_report()
