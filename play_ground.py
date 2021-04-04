# %%
from data_reader import get_trajectory
import config
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, rfftn, irfftn
from data_reader import read_one_file
from scipy import signal
from data_analyzer import cca_score
from data_analyzer import data_mean

# %%
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size / 2) - 10:]

def drawfft(y):
    yf = fft(y)
    x = fftfreq(len(y), 1/100)
    return x, yf

# %%
adam_e = get_trajectory(name='Adam', emotion='e')
adam_i = get_trajectory(name='Adam', emotion='i')
Bonn_e = get_trajectory(name='Bonn', emotion='e')
Bonn_i = get_trajectory(name='Bonn', emotion='i')

n = get_trajectory(emotion='n')
I = get_trajectory(emotion='i')
e = get_trajectory(emotion='e')

alll = get_trajectory()

# %%
plt.figure(figsize=(100, 10))
plt.plot(np.array(adam_e[0]['RVx']))

# %%
for i in adam_e:
    plt.plot(autocorr(np.array(i['RVx'])))
# %%
for i in adam_i:
    plt.plot(autocorr(np.array(i['RVx'])))

# %%
for i in Bonn_e:
    plt.plot(autocorr(np.array(i['RVx'])))
# %%
for i in Bonn_i:
    plt.plot(autocorr(np.array(i['RVx'])))

# %%
for i in n:
    plt.plot(autocorr(np.array(i['RVx'])))
# %%
for i in I:
    plt.plot(autocorr(np.array(i['RVx'])))
# %%
for i in e:
    plt.plot(autocorr(np.array(i['RVx'])))









# %%
for i in n[10]:
    plt.plot(autocorr(np.array(i['Tx'])))
# %%
for i in adam_i:
    plt.plot(autocorr(np.array(i['RVx'][:500])))

# %%
for i in Bonn_e:    
    plt.plot(autocorr(np.array(i['RVx'])))
# %%
for i in n:
    plt.plot(autocorr(np.array(i['RVx'][:500])))
# %%
for i in I:
    plt.plot(autocorr(np.array(i['RVx'][:500])))
# %%
for i in e:
    plt.plot(autocorr(np.array(i['RVx'][:500])))

# %%



x, yf = drawfft(np.array(n[9]['RVx']))
plt.plot(x, np.log10(np.abs(yf)))

# %%
plt.plot(x[:int(len(x) / 2)], np.log(np.abs(yf[:int(len(yf) / 2)])))



# %%
arrays = []

for i in n:
    x, yf = drawfft(np.array(i['RVx'][:24000]))
    arrays.append(yf)
    plt.plot(x[:int(len(x) / 2)], np.log(yf[:int(len(yf) / 2)]))
    # plt.xlim([0, 5])
    # plt.ylim([0, 500])
yfs = np.empty((len(arrays), len(arrays[0])), dtype='complex')

for i in range(len(n)):
    yfs[i, :] = arrays[i]

# %%
yfs.shape
yfn = np.average(yfs, axis=0)
yfn.shape
plt.plot(x[:int(len(x) / 2)], np.log(np.abs(yfn[:int(len(yfn) / 2)])))

# %%
a = np.log(np.real(yfn[:int(len(yf) / 2)]))
a[6000]
np.log(np.abs(yfn[12000]))

# %%
yfn[1223]
# %%
for i in e:
    x, yf = drawfft(np.abs(np.array(i['Tz'])))
    plt.plot(x, yf)
    plt.xlim([-5, 5])
    plt.ylim([-500, 500])
# %%
for i in I:
    x, yf = drawfft(np.abs(np.array(i['RVx'])))
    plt.plot(x, yf)
    plt.xlim([-5, 5])
    plt.ylim([-500, 500])
# %%
for i in adam_e:
    x, yf = drawfft(np.abs(np.array(i['RVx'])))
    plt.plot(x, yf)
    plt.xlim([-5, 5])
    plt.ylim([-500, 500])
# %%
for i in Bonn_e:
    x, yf = drawfft(np.abs(np.array(i['RVx'])))
    plt.plot(x, yf)
    plt.xlim([-5, 5])
    plt.ylim([-500, 500])
# %%

a = np.array(adam_e[0])[:1000, :6]
# %%
af = rfftn(a)
# %%
af.shape
# %%
ia = irfftn(af)
# %%
ia
# %%
np.max(af)
# %%









import pywt
# %%
widths = np.arange(1, 100)
cwtmatr, freqs = pywt.cwt(np.array(n[0]['RVx']), widths, 'mexh')
plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()
# %%


d = read_one_file('/home/tai/Desktop/results_stable.axa')
# %%
x, yf = drawfft(np.abs(np.array(d['Nx'])))
plt.plot(x, yf)
plt.xlim([-5, 5])
plt.ylim([-200, 200])


# %%
y = alll[0]['RVx']
x = alll[0]['RVy']
f, Cxy = signal.coherence(x, y, 100, nperseg=1024)
plt.semilogy(f, Cxy)

# %%
from sklearn.cross_decomposition import CCA
# %%
data = get_trajectory(name='Adam', seq=1)
X = data[0].iloc[:1000, :6]
Y = data[0].iloc[10000:11000, :6]
R = np.random.rand(1000, 6) - 1 / 2


#%%
X = (X - X.mean()) / X.std()
Y = (Y - Y.mean()) / Y.std()

print(X.shape)
print(Y.shape)

# %%
ca = CCA(n_components=6)
ca.fit(X, Y)
Xc, Yc = ca.transform(X, Y)

#%%
from data_analyzer import cca_score

#%%
print(cca_score(X, Y))
# %%
print(Xc.shape)
print(Yc.shape)
# %%
for i in range(6):
    print(np.corrcoef(Xc[:, i], Yc[:, i]))

# %%
ca = CCA(n_components=1)
ca.fit(X, R)
Xc, Rc = ca.transform(X, R)
# %%
print(Xc.shape)
print(Rc.shape)
# %%
for i in range(6):
    print(np.corrcoef(Xc[:, i], Rc[:, i]))
# %%
print(np.corrcoef(Xc[:, 0], Rc[:, 0]))

# %%
a = np.array([[1, 2], [1, 2]])
b = a

c = np.concatenate([a, b, a, b], axis=0)
d = np.concatenate((a, b), axis=1)
print(c)
print(d)
# %%
c
# %%
c.mean(axis=1)
# %%
print(config.DATA_MEAN)
# %%
data = get_trajectory()
m = data_mean(data)
# %%
print(m)
print(config.DATA_MEAN)

# %%
import config
from data_processor import standardize_data
from data_processor import trim_data
from data_reader import get_trajectory
from data_analyzer import data_length
# %%
data = get_trajectory()
data[0]
# %%
data = config.ALL_TRAJECTORY
data[0][3]
# %%
standardize_data()
data = config.ALL_TRAJECTORY
data[0][3]
# %%
type(config.ALL_TRAJECTORY)
# %%
print(config.STANDARDIZED)
# %%
print(config.DATA_MEAN)
# %%
data = get_trajectory('Adam')
# %%
len(data)
# %%
data = trim_data(data, length=1000)
# %%
data
# %%
len(data)
# %%
print(data_length(data))

# %%
import numpy as np
import config
from data_processor import standardize_data
from data_processor import trim_data
from data_reader import get_trajectory
from data_analyzer import data_length
from data_analyzer import average_cca_score
from data_analyzer import cca_score
# %%
data = get_trajectory(name='Adam')
data = trim_data(data)
R = np.random.rand(1000, 6) - 1 / 2
# %%
len(data)
# %%
print(average_cca_score(R, data))
print(cca_score(data[0], R))
# %%
config.DATA_FORMAT
# %%
print(data_length(data))
# %%
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
# %%
data = get_trajectory('Adam', seq=1)[0].iloc[:, 0]
# %%
data
# %%
f, t, sxx = spectrogram(data, fs=100)
plt.pcolormesh(t, f[:50], sxx[:50, :], shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [Sec]')
plt.show()
# %%
sxx.shape
# %%
t.shape
# %%
f.shape
# %%
f
# %%
a = np.array([3 + 4j, 2 + 2j, 3 - 4j, -4 + 3j])
# %%
a
# %%
np.absolute(a)
# %%
import numpy as np
import config
from data_reader import get_trajectory
from data_processor import standardize_data
from data_processor import pca_data
from data_processor import ipca_data
from data_analyzer import data_pca
# %%
data = standardize_data()
data_pca(n_cp=4)
data = pca_data(data)
# %%
config.PCA.explained_variance_ratio_


# %%
data = ipca_data(data)
# %%
data[0].shape
# %%
print(config.PCA.n_components_)
# %%
from scipy.fft import fft, fftfreq
xf = fftfreq(51, 1/51)
# %%
xf
# %%
data = get_trajectory('Adam')
# %%
data = np.array(data)
# %%
data = fft(data)
# %%
data.dtype
# %%
from data_processor import fft_data
from data_processor import ifft_data
import numpy as np
import config
from data_reader import get_trajectory
from data_processor import trim_data
from scipy.fft import fft, fftfreq, ifft

# %%
data = get_trajectory('Adam')
data = trim_data(data, 300)
# %%
data[0]
# %%
data_f = fft_data(data)
# %%
data_f[0].shape

# %%
idata = ifft_data(data_f)
# %%
idata[0]
# %%
data[0] - idata[0]
# %%
x = data[0].iloc[:, 0]
# %%
x
# %%
fx = fft(np.array(x))
# %%
x
# %%
pfx = fx[:151]
pfx.shape

# %%
nfx = np.conj(np.flip(pfx[1:150]))
# %%
nfx.shape
# %%
fx.shape
# %%
cfx = np.concatenate([pfx, nfx])

# %%
cfx.shape
# %%
fx - cfx
# %%
cfx.imag[-10:]
# %%
fx.imag[-10:]
# %%
fx + cfx

# %%
from data_processor import fft_data
from data_processor import ifft_data
import numpy as np
import config
from data_reader import get_trajectory
from data_processor import trim_data
from scipy.fft import fft, fftfreq, ifft
from data_analyzer import spectra_diff
from data_processor import fft_all_data
from data_analyzer import average_spectra_diff
from config import set_trim_length
from data_processor import standardize_all_data

# %%
set_trim_length(300)
data = get_trajectory('Adam')
data = trim_data(data)
standardize_data()
fft_all_data()

# %%
dataf = fft_data(data)
X = dataf[0]
Y = dataf[5]
# %%
X
# %%
Y
# %%
spectra_diff(X, Y)
# %%
R = [np.zeros((300, 6))]
Rf = fft_data(R)
# %%
spectra_diff(Y, Rf[0])
# %%
print(config.TRIM_LENGTH)
# %%
config.AVERAGE_SPECTRA
# %%
data = trim_data(data)

# %%
average_spectra_diff(Rf[0])
# %%
R[0].shape
# %%
adiff
# %%
type(Rf[0])
# %%
np.array(Rf).shape
# %%
Rf[0]
# %%

# %%

# %%
print(56)
# %%
import numpy as np
import matplotlib.pyplot as plt
from data_processor import istandardize_data
from data_processor import standardize_all_data
from data_reader import get_trajectory
from data_processor import fft_all_data
from data_processor import ifft_data
from config import set_trim_length
import config
from data_processor import trim_data
from data_processor import pca_data
from data_processor import ipca_data
from data_analyzer import average_spectra_diff
from data_analyzer import spectra_diff
# %%
standardize_all_data()
set_trim_length(300)
dataf = fft_all_data()
# %%
smallest = 1
for i in range(1000):
    x = dataf[i]
    score = average_spectra_diff(x)
    if score <= smallest:
        smallest = score
print(smallest)
# %%
for i in range(20):
    x = dataf[i]
    y = dataf[i + 100]
    r = np.random.randn(151, 6)
    z = np.zeros_like(x)
    print(average_spectra_diff(r))
# %%
dataf[0].shape[1]
# %%
x = np.absolute(dataf[0])
# %%
x.shape[1]
# %%
plt.plot(x)
# %%
x.dtype
# %%
a = np.absolute(x)
# %%
a.dtype
# %%
smallest
# %%
i
# %%
path = config.DATA_PATH
# %%
path
# %%
import os
# %%
p = os.path.join(path, 'Trajectory_Data', 'Normalised', 'Adam', 'Adam_01_n.rov')
# %%
p
# %%
f = open(p, 'r')
# %%
f
# %%
content = []
for line in f:
    content.append(line)
# %%
print(content[0])
# %%
content[0]
# %%
header = ''
for i in range(16):
    header += content[i]
# %%
header
# %%
print(header)
# %%
data = get_trajectory()
# %%
print(config.FORMAT_HEADER)
# %%
print(config.HEADERS[config.DATA_FORMAT])
# %%
get_trajectory(format='qtn')
# %%
from data_reader import get_trajectory
from data_reader import write_one_file
import os
import numpy as np
import config
# %%
data = get_trajectory()
x = data[3]
path = os.path.join(config.DATA_PATH, 'Generated', 'example.rov')
write_one_file(path, x, 'rov')
# %%
np.zeros()

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
x = np.linspace(0, 100, 400)
y1 = x * 3 - 300
y2 = x * (-3) + 200
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(x, y1, label='legend')
ax1.plot(x, y2, label='haha')
ax1.legend()
ax1.set_xlabel('awef', fontsize=14)
ax1.set_ylabel('awefd', fontsize=14)
ax2.plot(x, y2)
ax1.set_title('This is a graph', fontsize=18)
ax2.set_title('This is another one')

# %%
fig.savefig('/home/tai/Desktop/example.png', )

# %%
np.arange(100, 100 * 15 + 1, 100)
# %%
a = [[1.3, 1.2], [1.3, 1.2], [0.3, 0.2]]
# %%
a = np.array(a)
# %%
a.shape
# %%
a = a.T
a[0]
# %%
a.shape
# %%
a = a.T
# %%
a.shape
# %%
a[1]
# %%
a = np.array([])
# %%
a
# %%
a.shape
# %%
from dynamic_reporter import init_dynamic_report
from dynamic_reporter import stop_dynamic_report
import dynamic_reporter
import time
# %%
q = init_dynamic_report(10)
# %%
stop_dynamic_report(564)
# %%
q
# %%
q.qsize()
# %%
q.put({'wefwf': 58789874987.456})
# %%
def f():
    global A
    if 'A' in globals():
        print('ok')
    else:
        print('no')
# %%
f()
# %%
A = 3
# %%
import torch
import numpy as np

# %%
a = torch.tensor([1, 2.])
# %%
a
# %%
b = a.to(torch.device('cuda'))
# %%
b.requires_grad = True
# %%
b = (b * 3 + 4)**3
# %%
c = b.sum()
# %%
c
# %%
b
# %%
x = float(c.detach().cpu())
# %%
x
# %%
import torch
import numpy as np
from model_complex_fullconnected import Complex_Fully_Connected_GAN
# %%
gan = Complex_Fully_Connected_GAN(6)
gan.to(gan.device)
gan.in_cpu = False
# %%
torch.save(gan.state_dict(), '/home/tai/Desktop/example.model')
# %%
rgan = Complex_Fully_Connected_GAN(6)
rgan.load_state_dict(torch.load('/home/tai/Desktop/example.model'))
# %%
examples = rgan.generate(100)
# %%
examples
# %%
import datetime
# %%
'{:%Y-%m-%d|%H:%M:%S}'.format(datetime.datetime.now())
# %%
gan.in_cpu = False
# %%
gan.generate(1)
# %%
gan.in_cpu
# %%
rgan.generate(1)
# %%
rgan.in_cpu
# %%
def f(a, b):
    return a, b
# %%
f(*(1, 2, ))
# %%
import numpy as np
from play import play_one_video_from
from model_complex_fullconnected import Complex_Fully_Connected_GAN
from data_processor import iflatten_complex_data
from data_processor import ifft_data
import matplotlib.pyplot as plt
# %%
net = play_one_video_from(Complex_Fully_Connected_GAN, '/home/tai/UG4_Project/training_system/BEST_ASD', args=(6, ))
# %%
spec = net.generate(1)
# %%
data = iflatten_complex_data(net.generate(1).detach())
# %%
spec
# %%
plt.plot(np.absolute(data[0]))
# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
x = np.linspace(0, 100, 400)
y1 = x * 3 - 300
y2 = x * (-3) + 200
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(x, y1, label='legend')
ax1.plot(x, y2, label='haha')
ax1.legend()
ax1.set_xlabel('awef', fontsize=14)
ax1.set_ylabel('awefd', fontsize=14)
ax2.plot(x, y2)
ax1.set_title('This is a graph', fontsize=18)
ax2.set_title('This is another one')

# %%
plt.close(fig)
# %%
a = np.array([3 + 4j])
# %%
a
# %%
a.real
# %%
from data_reader import get_trajectory
from data_analyzer import average_cca_score
from data_processor import trim_data
import random
# %%
data = trim_data(get_trajectory(), 300)
# %%
a = average_cca_score(np.ones((300, 6)), random.choices(data, k=10))
# %%
a
# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
x = np.linspace(0, 100, 400)
y1 = x * 3 - 300
y2 = x * (-3) + 200
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
ax1.plot(x, y1, label='legend')
ax1.plot(x, y2, label='haha')
ax1.legend()
ax1.set_xlabel('awef', fontsize=14)
ax1.set_ylabel('awefd', fontsize=14)
ax2.plot(x, y2)
ax2.set_ylabel('awefawef')
ax1.set_title('This is a graph', fontsize=18)
ax2.set_title('This is another one')
ax3.set_title('awfawef')
ax4.clear()
# %%
from data_analyzer import average_spectra_diff
import numpy as np
# %%
a = np.array([0, 1, 2, 3])
# %%
b = np.tanh(a)
# %%
1 - b
# %%
from play import play_one_video_from
from play import play_long_video_from
from model_complex_fully_connected_wgan import Complex_Fully_Connected_WGAN
from model_complex_fully_connected_adjust import Complex_Fully_Connected_Adjust_GAN
from model_complex_fullconnected import Complex_Fully_Connected_GAN
from model_complex_fully_connected_wgan_pca import Complex_Fully_Connected_WGAN_PCA

# %%
Choosed_MLP = '/home/tai/UG4_Project/Data/Trained_Models/Complex_Fully_Connected/2021-03-16|08:34:22|LAST|BC:10|g_eta:1e-05|d_eta:1e-05'
Choosed_WGAN = '/home/tai/UG4_Project/Data/Trained_Models/Complex_Fully_Connected_WGAN/2021-03-17|06:11:48|LAST|BC:10|g_eta:1e-05|d_eta:1e-05|n_critic:10|clip_value:0.01'
Choosed_PCA = '/home/tai/UG4_Project/Data/Trained_Models/Complex_Fully_Connected_WGAN_PCA/2021-03-17|19:14:13|LAST|BC:10|g_eta:0.0001|d_eta:0.0001|n_critic:10|clip_value:0.01'

# %%
play_one_video_from(Complex_Fully_Connected_WGAN_PCA, Choosed_PCA, args=(4, ), translation=True)

# %%
play_long_video_from(Complex_Fully_Connected_WGAN, Choosed_WGAN, 3000, args=(6, ), translation=True)
# %%
from data_analyzer import average_spectra_diff_score
from data_analyzer import average_cca_score
from data_analyzer import average_spectra_cca_score
from data_processor import standardize_all_data
from data_processor import fft_all_data
from data_processor import trim_data
from config import set_trim_length
import numpy as np
import matplotlib.pyplot as plt
import random
from data_analyzer import average_spectra
import matplotlib.pyplot as plt
# %%
set_trim_length(300)
# %%
fdata = fft_all_data()
data = trim_data(standardize_all_data())
# %%
len(fdata)
# %%
fdata[0].shape
# %%
scores = np.zeros((12709, ))
# %%
scores.shape
# %%
for i in range(len(data)):
    try:
        scores[i] = average_cca_score(data[i], random.choices(data, k=10))
    except:
        pass
# %%
plt.scatter(x=np.linspace(0, 12709, 12709), y=scores)
# %%
scores.mean()
# %%
avg = average_spectra()
# %%
plt.plot(np.absolute(fdata[786]))
# %%
import numpy as np
# %%
decrease = np.reshape(np.linspace(1, 0, num=150), (150, 1))

# %%
increase = 1 - decrease
# %%
decrease.shape
# %%
a = np.ones([150, 6], dtype=np.float32)
# %%
a * decrease + a * increase
# %%
decrease * a
# %%
a
# %%
a[0:-147]
# %%
a[-147:].shape
# %%
import torch
# %%
a = torch.tensor([[1, 2, -3.], [.13, 23, 1]])
# %%
b = torch.log(a)
# %%
b
# %%
type(b.size()[0])
# %%a

# %%
a
# %%
a[:, -2:] * b[:, :2]
# %%
from data_processor import fft_all_data
from config import set_trim_length
from play import random_log_spetra_from
from data_processor import fft_data
# %%
set_trim_length(300)
data = fft_all_data()
# %%
random_log_spetra_from(data)

# %%
from play import draw_spectra_of
from play import draw_log_spectra_of
from play import draw_trajectory_of
from data_processor import low_pass_filter
import matplotlib.pyplot as plt
from data_reader import write_one_file

# %%
draw_spectra_of('/home/tai/Desktop/Examples/WGAN-Gesture1.rov')
# %%
draw_log_spectra_of('/home/tai/Desktop/Examples/WGAN-Stable1.rov')
# %%
x = draw_trajectory_of('/home/tai/Desktop/Examples/MLP-Gesture4.rov')

# %%
y = low_pass_filter(x)
plt.plot(y)
# %%
write_one_file('/home/tai/Desktop/Examples/MLP-Gesture4-lpf.rov', y)
# %%
from scipy import signal
import matplotlib.pyplot as plt

# %%
a = signal.get_window('hamming', 300)
# %%
plt.plot(a)
# %%
a.shape
# %%
import numpy as np
# %%
x = [np.ones((300, 6), dtype=float)]
# %%
fx = fft_data(x)
# %%
fx
# %%
l = np.linspace(0, 1, 300)
# %%
l = np.reshape(l, (300, 1))
# %%
x[0] = x[0] * l
# %%
x[0] = x[0] * np.reshape(a, (300, 1))
# %%
plt.plot(np.log(fx[0]))
# %%
