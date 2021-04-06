# %%
from importlib import reload
import play
reload(play)
from play import play_one_video_from
from play import play_long_video_from
from play import play_long_video_istft
from play import play_rnn_OLA
from play import play_rnn_istft
from model_complex_fully_connected_wgan import Complex_Fully_Connected_WGAN
from model_complex_fully_connected_adjust import Complex_Fully_Connected_Adjust_GAN
from model_complex_fullconnected import Complex_Fully_Connected_GAN
from model_complex_fully_connected_wgan_pca import Complex_Fully_Connected_WGAN_PCA
from model_complex_fully_connected_wgan_lpf import Complex_Fully_Connected_WGAN_LPF
from model_wgan_lpf_window import Complex_Fully_Connected_WGAN_LPF_W
from model_wgan_lpf_window_adjusted import Complex_Fully_Connected_WGAN_LPF_W_Adjusted
from model_crelu_rnn_lpf_stft_wgan import Crelu_RNN_WGAN_LPF_W
from model_crelu_lpf_w import Crelu_Fully_Connected_WGAN_LPF_W
from model_wgan_lpf_window import WIN
from play import play_real
from Simple_GAN import Simple_GAN

# %%
Choosed_MLP = '/home/tai/UG4_Project/Data/Trained_Models/Complex_Fully_Connected/2021-03-16|08:34:22|LAST|BC:10|g_eta:1e-05|d_eta:1e-05'
Choosed_WGAN = '/home/tai/UG4_Project/Data/Trained_Models/Complex_Fully_Connected_WGAN/2021-03-17|06:11:48|LAST|BC:10|g_eta:1e-05|d_eta:1e-05|n_critic:10|clip_value:0.01'
Choosed_WGAN2 = '/home/tai/UG4_Project/Data/Trained_Models/Complex_Fully_Connected_WGAN/2021-03-17|08:01:16|LAST|BC:10|g_eta:0.0001|d_eta:0.0001|n_critic:10|clip_value:0.01'
Choosed_PCA = '/home/tai/UG4_Project/Data/Trained_Models/Complex_Fully_Connected_WGAN_PCA/2021-03-17|19:14:13|LAST|BC:10|g_eta:0.0001|d_eta:0.0001|n_critic:10|clip_value:0.01'
Choosed_lpf = '/home/tai/UG4_Project/Data/Trained_Models/Complex_Fully_Connected_WGAN_LPF/2021-03-27|09:42:28|LAST|BC:10|g_eta:0.001|d_eta:0.001|n_critic:5|clip_value:0.01'
Choosed_lpfw = '/home/tai/UG4_Project/Data/Trained_Models/Complex_Fully_Connected_WGAN_LPF_W/2021-04-04|23:26:52|LAST|BC:10|g_eta:0.0001|d_eta:0.0001|n_critic:5|clip_value:0.01'
Choosed_lpfwa = '/home/tai/UG4_Project/Data/Trained_Models/Complex_Fully_Connected_WGAN_LPF_W_Adjusted/2021-04-05|07:26:14|LAST|BC:10|g_eta:0.0001|d_eta:0.0001|n_critic:5|clip_value:0.01'
Choosed_lpfwa2 = '/home/tai/UG4_Project/Data/Trained_Models/Complex_Fully_Connected_WGAN_LPF_W_Adjusted/2021-04-05|08:37:50|LAST|BC:10|g_eta:0.0001|d_eta:0.0001|n_critic:5|clip_value:0.01'
Choosed_RNN = '/home/tai/UG4_Project/Data/Trained_Models/Crelu_RNN_WGAN_LPF_W/2021-04-06|21:06:48|LAST|BC:10|g_eta:0.0001|d_eta:0.0001|n_critic:5|clip_value:0.01'
Choosed_CRELU = '/home/tai/UG4_Project/Data/Trained_Models/Crelu_Fully_Connected_WGAN_LPF_W/2021-04-06|20:20:44|LAST|BC:10|g_eta:1e-05|d_eta:1e-05|n_critic:5|clip_value:0.01'
Choosed_Simple = '/home/tai/UG4_Project/Data/Trained_Models/Simple_GAN/2021-04-07|02:39:21|LAST|BC:10|g_eta:1e-05|d_eta:1e-05'

# %%
play_long_video_from(Complex_Fully_Connected_GAN, Choosed_MLP, 6000, args=(6, ), translation=True)

# %%
if __name__ == "__main__":
    play_long_video_from(Complex_Fully_Connected_WGAN_PCA, Choosed_PCA, 3000, args=(4, ), translation=True)
# %%
play_long_video_from(Complex_Fully_Connected_WGAN_LPF, Choosed_lpf, 6000, args=(31, ), translation=True)



# %%
play_long_video_from(Complex_Fully_Connected_WGAN_LPF_W_Adjusted, Choosed_lpfwa, 6000, args=(31, ), translation=True)
# %%
play_long_video_istft(Complex_Fully_Connected_WGAN_LPF_W_Adjusted, Choosed_lpfwa, 30, WIN, args=(31, ), translation=True, lpf=False)




# %%
play_long_video_from(Complex_Fully_Connected_WGAN_LPF_W_Adjusted, Choosed_lpfwa2, 6000, args=(31, ), translation=True)
# %%
play_long_video_istft(Complex_Fully_Connected_WGAN_LPF_W_Adjusted, Choosed_lpfwa2, 30, WIN, args=(31, ), translation=True)







# %%
play_long_video_from(Complex_Fully_Connected_WGAN, Choosed_WGAN, 6000, args=(6, ), translation=True)

# %%
a = play_long_video_istft(Complex_Fully_Connected_WGAN_LPF_W, Choosed_lpfw, 20, WIN, args=(31, ), translation=True)

# %%
WIN.shape
# %%
a.shape
# %%
a = play_long_video_istft(Complex_Fully_Connected_WGAN_LPF_W_Adjusted, Choosed_lpfwa, 20, WIN, args=(31, ), translation=True)

# %%
play_long_video_from(Complex_Fully_Connected_WGAN_LPF_W_Adjusted, Choosed_lpfwa, 6000, args=(31, ), translation=True)

# %%
x = play_rnn_OLA(Crelu_RNN_WGAN_LPF_W, Choosed_RNN, 30, WIN, 'rov', (31, ))
# %%
x[0].shape
# %%
len(x)
# %%
x = play_rnn_istft(Crelu_RNN_WGAN_LPF_W, Choosed_RNN, 30, WIN, 'rov', (31, ))
# %%
x = play_rnn_OLA(Crelu_RNN_WGAN_LPF_W, Choosed_RNN, 30, WIN, 'rov', (31, ))
# %%
play_long_video_from(Crelu_Fully_Connected_WGAN_LPF_W, Choosed_CRELU, 6000, args=(31, ), translation=True)

# %%
play_long_video_istft(Crelu_Fully_Connected_WGAN_LPF_W, Choosed_CRELU, 30, WIN, args=(31, ), translation=True)

# %%
play_real(4000)
# %%
import config
config.HEADERS['rov']
# %%
play_long_video_from(Simple_GAN, Choosed_Simple, 6000, 'rov', (6,), translation=True, lpf=False)

# %%
