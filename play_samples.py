# %%
from play import play_one_video_from
from play import play_long_video_from
from model_complex_fully_connected_wgan import Complex_Fully_Connected_WGAN
from model_complex_fully_connected_adjust import Complex_Fully_Connected_Adjust_GAN
from model_complex_fullconnected import Complex_Fully_Connected_GAN
from model_complex_fully_connected_wgan_pca import Complex_Fully_Connected_WGAN_PCA
from model_complex_fully_connected_wgan_lpf import Complex_Fully_Connected_WGAN_LPF
from model_wgan_lpf_window import Complex_Fully_Connected_WGAN_LPF_W

# %%
Choosed_MLP = '/home/tai/UG4_Project/Data/Trained_Models/Complex_Fully_Connected/2021-03-16|08:34:22|LAST|BC:10|g_eta:1e-05|d_eta:1e-05'
Choosed_WGAN = '/home/tai/UG4_Project/Data/Trained_Models/Complex_Fully_Connected_WGAN/2021-03-17|06:11:48|LAST|BC:10|g_eta:1e-05|d_eta:1e-05|n_critic:10|clip_value:0.01'
Choosed_PCA = '/home/tai/UG4_Project/Data/Trained_Models/Complex_Fully_Connected_WGAN_PCA/2021-03-17|19:14:13|LAST|BC:10|g_eta:0.0001|d_eta:0.0001|n_critic:10|clip_value:0.01'
Choosed_lpf = '/home/tai/UG4_Project/Data/Trained_Models/Complex_Fully_Connected_WGAN_LPF/2021-03-27|09:42:28|LAST|BC:10|g_eta:0.001|d_eta:0.001|n_critic:5|clip_value:0.01'
Choosed_lpfw = '/home/tai/UG4_Project/Data/Trained_Models/Complex_Fully_Connected_WGAN_LPF_W/2021-03-27|13:53:35|LAST|BC:10|g_eta:1e-05|d_eta:1e-05|n_critic:5|clip_value:0.01'

# %%
if __name__ == "__main__":
    play_one_video_from(Complex_Fully_Connected_WGAN_PCA, Choosed_PCA, args=(4, ), translation=True)

# %%
if __name__ == "__main__":
    play_long_video_from(Complex_Fully_Connected_WGAN_PCA, Choosed_PCA, 3000, args=(4, ), translation=True)
# %%
if __name__ == "__main__":
    play_long_video_from(Complex_Fully_Connected_WGAN_LPF_W, Choosed_lpfw, 3000, args=(16, ), translation=True)
# %%
play_long_video_from(Complex_Fully_Connected_WGAN_LPF_W, Choosed_lpfw, 3000, args=(16, ), translation=False)

# %%
