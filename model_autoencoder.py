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

CODE_LENGTH = 100
# Prepare the training set for this model
print('Preparing the training set...')
if config.TRIM_LENGTH is None:
    set_trim_length(1000)
origin = trim_data(standardize_all_data())
data = fft_all_data()
train_set = flatten_complex_data(data)
print('Training set is ready!')

def get_random_example(n, trim_length, trimed=True, standardize=True):
    if standardize:
        data = standardize_all_data()
    else:
        data = get_trajectory()

    if trimed:
        data = trim_data(data, length=trim_length)

    examples = random.choices(data, k=n)

    return examples

class Encoder(nn.Module):
    def __init__(self, dimension):
        super(Encoder, self).__init__()
        global CODE_LENGTH
        self.n_in = dimension * (config.TRIM_LENGTH // 2 + 1) * 2   # real part and imaginary part are saperated

        # hidden linear layers
        self.linear1 = nn.Linear(self.n_in, self.n_in)
        self.linear2 = nn.Linear(self.n_in, self.n_in)
        self.linear3 = nn.Linear(self.n_in, self.n_in)
        self.linear4 = nn.Linear(self.n_in, CODE_LENGTH)

        self.criterion = nn.BCELoss()

        return

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), 0.1)
        x = F.leaky_relu(self.linear2(x), 0.1)
        x = F.leaky_relu(self.linear3(x), 0.1)
        x = self.linear4(x)

        return x


class Decoder(nn.Module):
    def __init__(self, dimension):
        super(Decoder, self).__init__()
        global CODE_LENGTH
        self.n_in = CODE_LENGTH
        self.n_out = dimension * (config.TRIM_LENGTH // 2 + 1) * 2

        # linear layers
        self.linear1 = nn.Linear(self.n_in, self.n_out)
        self.linear2 = nn.Linear(self.n_out, self.n_out)
        self.linear3 = nn.Linear(self.n_out, self.n_out)
        self.linear4 = nn.Linear(self.n_out, self.n_out)

        return

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), 0.1)
        x = F.leaky_relu(self.linear2(x), 0.1)
        x = F.leaky_relu(self.linear3(x), 0.1)
        out = self.linear4(x)

        return out


class Autoencoder(nn.Module):
    def __init__(self, dimension):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(dimension)
        self.decoder = Decoder(dimension)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_cpu = True

        # Variables to record training process
        self.losses = []
        self.scores = []

        return

    def forward(self, x):
        code = self.encoder(x)

        return self.decoder(code)

    def decode(self, x):
        return self.decoder(x)

    def encode(self, x):
        return self.encoder(x)

    def example(self):
        r = get_random_example(1, config.TRIM_LENGTH)
        e = fft_data(r)
        e = flatten_complex_data(e)
        if self.in_cpu:
            e = torch.tensor(e)
        else:
            e = torch.tensor(e, device=self.device, dtype=torch.float32)

        e = self.forward(e).detach().cpu()
        e = np.array(e)

        e = iflatten_complex_data(e)
        e = ifft_data(e)[0]
        r = np.array(r[0])

        return e - r

    def train(self, train_set, batch_size, num_epoche, e_eta, d_eta, show=True):
        print('Start training | batch_size:{a} | e_eta:{b} | d_eta:{c}'.format(a=batch_size, b=e_eta, c=d_eta))
        self.to(self.device)
        self.in_cpu = False
        train_set = torch.tensor(train_set, dtype=torch.float, device=self.device)

        e_optimizer = optim.Adam(self.encoder.parameters(), lr=e_eta)
        d_optimizer = optim.Adam(self.decoder.parameters(), lr=d_eta)
        mse = nn.MSELoss(reduction='mean')

        N = train_set.size()[0]
        N = N - N % batch_size

        lowest_loss = sys.float_info.max

        for epoch in range(num_epoche):
            tic = time.time()

            perm = torch.randperm(N)

            steps = 0
            for i in range(0, N, batch_size):
                e_optimizer.zero_grad()
                d_optimizer.zero_grad()

                indices = perm[i:i + batch_size]

                real = train_set[indices]
                fake = self.forward(real)

                loss = mse(real, fake)

                loss.backward()
                e_optimizer.step()
                d_optimizer.step()

                steps += 1

                if show:
                    if steps % 100 == 0:
                        # Record training losses
                        loss = float(loss.detach().cpu())
                        self.losses.append([loss])
                        self.scores.append([loss])

                        if loss < lowest_loss:
                            lowest_loss = loss
                            if epoch > 10:
                                torch.save(self.state_dict(), 'BEST_MSE')

                            report(
                                loss_title='Training loss curve',
                                losses=self.losses,
                                loss_labels=['MSE Loss'],
                                score_title='Model score curve',
                                scores=self.scores,
                                score_labels=['MSE Loss'],
                                interval=100,
                                example=self.example()
                            )

            dt = time.time() - tic
            print('epoch ' + str(epoch) + '\tfinished! Time usage: ' + str(dt) + '\t Loss: ' + str(loss))

        self.to(torch.device('cpu'))
        self.in_cpu = True

        last_path = os.path.join(
            config.DATA_PATH,
            'Trained_Models',
            'Autoencoder',
            time_stamp() + '|LAST' + '|BC:' + str(batch_size) + '|e_eta:' + str(e_eta) + '|d_eta:' + str(d_eta)
        )
        torch.save(self.state_dict(), last_path)

        # Store the model with lowest loss
        try:
            self.load_state_dict(torch.load('BEST_MSE'))
            os.remove('BEST_MSE')

            best_asd_path = os.path.join(
                config.DATA_PATH,
                'Trained_Models',
                'Autoencoder',
                time_stamp() + '|MSE' + '|BC:' + str(batch_size) + '|e_eta:' + str(e_eta) + '|d_eta:' + str(d_eta)
            )
            torch.save(self.state_dict(), best_asd_path)
        except:
            pass

        return


if __name__ == '__main__':
    set_start_method('spawn')   # To make dynamic reporter works

    for eta in [0.00001, 0.0001]:
        for i in range(3):
            report_path = os.path.join(
                config.DATA_PATH,
                'Training_Reports',
                'Autoencoder',
                time_stamp() + '|eta:' + str(eta) + '.png'
            )
            init_dynamic_report(3, report_path)
            gan = Autoencoder(6)
            gan.train(train_set, 10, 100, eta, eta, True)
            stop_dynamic_report()
