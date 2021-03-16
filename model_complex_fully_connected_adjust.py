# This model uses a fully connected generator and a fully connected discriminator.
# The input are complex numbers, but saperated into real and imaginary part
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
from data_processor import fft_all_data
from data_processor import fft_data
from data_processor import ifft_data
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
import time

# Prepare the training set for this model
print('Preparing the training set...')
if config.TRIM_LENGTH is None:
    set_trim_length(300)
origin = trim_data(standardize_all_data())
data = fft_all_data()
train_set = flatten_complex_data(data)
print('Training set is ready!')

class Complex_Fully_Connected_Discriminator(nn.Module):
    def __init__(self, dimension):
        super(Complex_Fully_Connected_Discriminator, self).__init__()
        self.n_in = dimension * (config.TRIM_LENGTH // 2 + 1) * 2   # real part and imaginary part are saperated

        # hidden linear layers
        self.linear1 = nn.Linear(self.n_in, self.n_in)
        self.linear2 = nn.Linear(self.n_in, self.n_in)
        self.linear3 = nn.Linear(self.n_in, 1)
        self.drop_layer = nn.Dropout(0.87)

        self.criterion = nn.BCELoss()

        return

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), 0.1)
        x = self.drop_layer(x)
        x = F.leaky_relu(self.linear2(x), 0.1)
        x = self.drop_layer(x)
        x = torch.sigmoid(self.linear3(x))

        return x


class Complex_Fully_Connected_Generator(nn.Module):
    def __init__(self, dimension):
        super(Complex_Fully_Connected_Generator, self).__init__()
        self.n_in = 20
        self.n_out = dimension * (config.TRIM_LENGTH // 2 + 1) * 2

        # linear layers
        self.linear1 = nn.Linear(self.n_in, self.n_out * 2)
        self.linear2 = nn.Linear(self.n_out * 2, self.n_out * 2)
        self.linear3 = nn.Linear(self.n_out * 2, self.n_out * 2)
        self.linear4 = nn.Linear(self.n_out * 2, self.n_out)

        return

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), 0.1)
        x = F.leaky_relu(self.linear2(x), 0.1)
        x = F.leaky_relu(self.linear3(x), 0.1)
        out = self.linear4(x)

        return out


class Complex_Fully_Connected_Adjust_GAN(nn.Module):
    def __init__(self, dimension):
        super(Complex_Fully_Connected_Adjust_GAN, self).__init__()
        self.dimension = dimension
        self.generator = Complex_Fully_Connected_Generator(dimension)
        self.discriminator = Complex_Fully_Connected_Discriminator(dimension)

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
        x = self.generate(1).detach().cpu()
        x = np.array(x)
        x = iflatten_complex_data(x)
        x = ifft_data(x)[0]

        return x

    def noise(self, batch_size):
        if self.in_cpu:
            return torch.randn([batch_size, 20])

        return torch.randn([batch_size, 20], device=self.device)

    def train(self, train_set, batch_size, num_epoche, g_eta, d_eta, show=True):
        print('Start training | batch_size:{a} | eta:{b}'.format(a=batch_size, b=g_eta))
        global origin
        self.to(self.device)
        self.in_cpu = False
        train_set = torch.tensor(train_set, dtype=torch.float, device=self.device)

        g_optimizer = optim.Adam(self.generator.parameters(), lr=g_eta)
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=d_eta)

        g_target = torch.ones(batch_size, 1).to(self.device)
        d_target = torch.cat([torch.zeros(batch_size, 1), torch.ones(batch_size, 1)], 0).to(self.device)

        N = train_set.size()[0]
        N = N - N % batch_size

        best_asds_score = 0
        best_cca_score = 0
        best_scca_score = 0

        for epoch in range(num_epoche):
            tic = time.time()

            perm = torch.randperm(N)
            
            steps = 0
            for i in range(0, N, batch_size):
                # optimize generator
                g_optimizer.zero_grad()
                noise = self.noise(batch_size)
                g_out = self.generator(noise)
                gd_out = self.discriminator(g_out)

                g_loss = self.discriminator.criterion(gd_out, g_target)
                g_loss.backward()
                g_optimizer.step()

                # optimize discriminator
                d_optimizer.zero_grad()
                indices = perm[i:i + batch_size]
                d_input = torch.cat([g_out.detach(), train_set[indices]], 0)
                d_out = self.discriminator.forward(d_input)
                
                d_loss = self.discriminator.criterion(d_out, d_target)
                d_loss.backward()
                if g_loss < 3 * d_loss:
                    d_optimizer.step()

                steps += 1
                if show:
                    if steps % 100 == 0:
                        # record training losses
                        g_r = float(g_loss.detach().cpu())
                        d_r = float(d_loss.detach().cpu())
                        self.losses.append([g_r, d_r])

                        # record model score
                        with torch.no_grad():
                            fake = self.generate(1).cpu()
                        
                        # Average spectra diff score
                        fake = iflatten_complex_data(fake)
                        example = self.example()

                        score_asds = average_spectra_diff_score(fake[0])
                        try:
                            score_scca = average_spectra_cca_score(fake[0])
                        except:
                            score_scca = 0
                        try:
                            score_cca = average_cca_score(example, random.choices(origin, k=10))
                        except:
                            score_cca = 0

                        self.scores.append([score_asds, score_cca, score_scca])

                        if score_asds > best_asds_score:
                            best_asd_score = score_asds
                            if epoch > 0:
                                torch.save(self.state_dict(), 'BEST_ASDS')

                        if score_cca > best_cca_score:
                            best_cca_score = score_cca
                            if epoch > 0:
                                torch.save(self.state_dict(), 'BEST_CCA')

                        if score_scca > best_scca_score:
                            best_scca_score = score_scca
                            if epoch > 0:
                                torch.save(self.state_dict(), 'BEST_SCCA')

                        report(
                            loss_title='Training Loss Curve',
                            losses=self.losses,
                            loss_labels=['Generator', 'Discriminator'],
                            score_title='Model Score Curve',
                            scores=self.scores,
                            score_labels=['ASD', 'CCA', 'SCCA'],
                            interval=100,
                            example=example
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

        # Store the model with best ASD score
        try:
            self.load_state_dict(torch.load('BEST_ASDS'))
            os.remove('BEST_ASDS')

            best_asd_path = os.path.join(
                config.DATA_PATH,
                'Trained_Models',
                'Complex_Fully_Connected_Adjust',
                time_stamp() + '|ASDS' + '|BC:' + str(batch_size) + '|g_eta:' + str(g_eta) + '|d_eta:' + str(d_eta)
            )
            torch.save(self.state_dict(), best_asd_path)
        except:
            pass

        # Store the model with best CCA score
        try:
            self.load_state_dict(torch.load('BEST_CCA'))
            os.remove('BEST_CCA')

            best_cca_path = os.path.join(
                config.DATA_PATH,
                'Trained_Models',
                'Complex_Fully_Connected_Adjust',
                time_stamp() + '|CCA' + '|BC:' + str(batch_size) + '|g_eta:' + str(g_eta) + '|d_eta:' + str(d_eta)
            )
            torch.save(self.state_dict(), best_cca_path)
        except:
            pass

        # Store the model with best SCCA score
        try:
            self.load_state_dict(torch.load('BEST_SCCA'))
            os.remove('BEST_SCCA')

            best_scca_path = os.path.join(
                config.DATA_PATH,
                'Trained_Models',
                'Complex_Fully_Connected_Adjust',
                time_stamp() + '|SCCA' + '|BC:' + str(batch_size) + '|g_eta:' + str(g_eta) + '|d_eta:' + str(d_eta)
            )
            torch.save(self.state_dict(), best_scca_path)
        except:
            pass

        return


if __name__ == '__main__':
    set_start_method('spawn')   # To make dynamic reporter works

    for eta in [0.00001, 0.0001, 0.000001, 0.001]:
        for i in range(5):
            report_path = os.path.join(
                config.DATA_PATH,
                'Training_Reports',
                'Complex_Fully_Connected_Adjust',
                time_stamp() + '|eta:' + str(eta) + '.png'
            )
            init_dynamic_report(3, report_path)
            gan = Complex_Fully_Connected_Adjust_GAN(6)
            gan.train(train_set, 10, 300, eta, eta, True)
            stop_dynamic_report()
