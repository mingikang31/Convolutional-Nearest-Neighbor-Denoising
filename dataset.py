import torch
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from mnist1d.data import make_dataset
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

# Datasets from Data/utils.py
from Data.utils import NoisyBSD68, NoisyCBSD68, NoisyCIFAR10 


class Denoise_BSD68:
    def __init__(self, data_dir, batch_size, noise_std, train_count, target_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.train_count = train_count
        self.target_size = target_size

        # Test dataset and DataLoader
        self.train_dataset = NoisyBSD68(data_dir=data_dir, target_count=train_count, target_size=target_size, noise_std=noise_std)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Singular test image
        self.test_data = self.train_dataset.test_image()

    def shape(self):    
        return self.train_dataset[0][0].shape

class Denoise_CBSD68:
    def __init__(self, data_dir, batch_size, noise_std, train_count, target_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.train_count = train_count
        self.target_size = target_size

        # Test dataset and DataLoader
        self.train_dataset = NoisyCBSD68(data_dir=data_dir, target_count=train_count, target_size=target_size, noise_std=noise_std)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Singular test image
        self.test_data = self.train_dataset.test_image()

    def shape(self):    
        return self.train_dataset[0][0].shape


class Denoise_CIFAR10:
    def __init__(self, data_dir, batch_size, noise_std):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.noise_std = noise_std

        # Create the training dataset and DataLoader
        self.train_dataset = NoisyCIFAR10(root=data_dir, train=True, noise_std=noise_std)
        self.test_dataset = NoisyCIFAR10(root=data_dir, train=False, noise_std=noise_std)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def shape(self):
        return self.train_dataset[0][0].shape

'''MNIST 1D data for training 1D CNN Models'''

import torch
import numpy as np
# from mnist1d.data import make_dataset

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d



class MNIST1D_Dataset():
   
    def __init__(self, seed = None): 

      
        self.data_args = self.get_dataset_args(as_dict=False)

        self.data_args_dict = self.get_dataset_args(as_dict=True)

        self.model_args = self.get_model_args(as_dict=False)

        self.model_args_dict = self.get_model_args(as_dict=True)

        if not seed: 
            self.set_seed(self.data_args.seed)
        else: 
            self.set_seed(seed)
            
        # print("The Arguments for Data are: ")
        # print("num_samples: 5000 \n train_split: 0.8 \n template_len: 12 \n padding: [36,60] \n scale_coeff: .4 \n max_translation: 48 \n corr_noise_scale: 0.25 \n iid_noise_scale: 2e-2 \n shear_scale: 0.75 \n shuffle_seq: False \n final_seq_length: 40 \n seed: 42")

        # print("\n")

        # print("The Arguments for Model are: ")
        # print("input_size: 40 \n output_size: 10 \n hidden_size: 256 \n learning_rate: 1e-2 \n weight_decay: 0 \n batch_size: 100 \n total_steps: 6000 \n print_every: 1000 \n eval_every: 250 \n checkpoint_every: 1000 \n device: mps \n seed: 42")



    def make_dataset(self): 
        data = make_dataset(self.data_args)
        # Creating dataset of size [Batch, channels, tokens]
        data['x'] = torch.Tensor(data['x']).unsqueeze(1)
        data['x_test'] = torch.Tensor(data['x_test']).unsqueeze(1)
        return data

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_dataset_args(self, as_dict=False):
        arg_dict = {'num_samples': 5000,
                'train_split': 0.8,
                'template_len': 12,
                'padding': [36,60],
                'scale_coeff': .4, 
                'max_translation': 48,
                'corr_noise_scale': 0.25,
                'iid_noise_scale': 2e-2,
                'shear_scale': 0.75,
                'shuffle_seq': False,
                'final_seq_length': 40,
                'seed': 42}
        return arg_dict if as_dict else self.ObjectView(arg_dict)

    def get_model_args(self, as_dict=False):
        arg_dict = {'input_size': 40,
                'output_size': 10,
                'hidden_size': 256,
                'learning_rate': 1e-2,
                'weight_decay': 0,
                'batch_size': 100,
                'total_steps': 6000,
                'print_every': 1000,
                'eval_every': 250,
                'checkpoint_every': 1000,
                'device': 'mps',
                'seed': 42}
        return arg_dict if as_dict else self.ObjectView(arg_dict)

    @staticmethod
    class ObjectView(object):
        def __init__(self, d): self.__dict__ = d
      
class MNIST1D_Plot(): 
    def __init__(self, data=None, data_args=None):
        self.data = data
        self.data_args = data_args
        

    '''Functions for Transformation'''
    def pad(self, x, padding): 
        low, high = padding
        p = low + int(np.random.rand()*(high-low+1))
        return np.concatenate([x, np.zeros((p))])

    def shear(self, x, scale=10):
        coeff = scale*(np.random.rand() - 0.5)
        return x - coeff*np.linspace(-0.5,.5,len(x))

    def translate(self, x, max_translation):
        k = np.random.choice(max_translation)
        return np.concatenate([x[-k:], x[:-k]])

    def corr_noise_like(self, x, scale):
        noise = scale * np.random.randn(*x.shape)
        return gaussian_filter(noise, 2)

    def iid_noise_like(self, x, scale):
        noise = scale * np.random.randn(*x.shape)
        return noise

    def interpolate(self, x, N):
        scale = np.linspace(0,1,len(x))
        new_scale = np.linspace(0,1,N)
        new_x = interp1d(scale, x, axis=0, kind='linear')(new_scale)
        return new_x

    def transform(self, x, y, args, eps=1e-8):
        new_x = self.pad(x+eps, args.padding) # pad
        new_x = self.interpolate(new_x, args.template_len + args.padding[-1])  # dilate
        new_y = self.interpolate(y, args.template_len + args.padding[-1])
        new_x *= (1 + args.scale_coeff*(np.random.rand() - 0.5))  # scale
        new_x = self.translate(new_x, args.max_translation)  #translate
        
        # add noise
        mask = new_x != 0
        new_x = mask*new_x + (1-mask)*self.corr_noise_like(new_x, args.corr_noise_scale)
        new_x = new_x + self.iid_noise_like(new_x, args.iid_noise_scale)
        
        # shear and interpolate
        new_x = self.shear(new_x, args.shear_scale)
        new_x = self.interpolate(new_x, args.final_seq_length) # subsample
        new_y = self.interpolate(new_y, args.final_seq_length)
        return new_x, new_y


    '''Additional Functions for plotting'''
    def apply_ablations(self, arg_dict, n=7): 
        ablations = [('shear_scale', 0),
                    ('iid_noise_scale', 0),
                    ('corr_noise_scale', 0),
                    ('max_translation', 1),
                    ('scale_coeff', 0),
                    ('padding', [arg_dict['padding'][-1], arg_dict['padding'][-1]]),
                    ('padding', [0, 0]),]
        num_ablations = min(n, len(ablations))
        for i in range(num_ablations):
            k, v = ablations[i]
            arg_dict[k] = v
        return arg_dict

    def get_templates(self):
        d0 = np.asarray([5,6,6.5,6.75,7,7,7,7,6.75,6.5,6,5])
        d1 = np.asarray([5,3,3,3.4,3.8,4.2,4.6,5,5.4,5.8,5,5])
        d2 = np.asarray([5,6,6.5,6.5,6,5.25,4.75,4,3.5,3.5,4,5])
        d3 = np.asarray([5,6,6.5,6.5,6,5,5,6,6.5,6.5,6,5])
        d4 = np.asarray([5,4.4,3.8,3.2,2.6,2.6,5,5,5,5,5,5])
        d5 = np.asarray([5,3,3,3,3,5,6,6.5,6.5,6,4.5,5])
        d6 = np.asarray([5,4,3.5,3.25,3,3,3,3,3.25,3.5,4,5])
        d7 = np.asarray([5,7,7,6.6,6.2,5.8,5.4,5,4.6,4.2,5,5])
        d8 = np.asarray([5,4,3.5,3.5,4,5,5,4,3.5,3.5,4,5])
        d9 = np.asarray([5,4,3.5,3.5,4,5,5,5,5,4.7,4.3,5])

        x = np.stack([d0,d1,d2,d3,d4,d5,d6,d7,d8,d9])
        x -= x.mean(1,keepdims=True) # whiten
        x /= x.std(1,keepdims=True)
        x -= x[:,:1]  # signal starts and ends at 0

        templates = {'x': x/6., 't': np.linspace(-5, 5, len(d0))/6.,
                'y': np.asarray([0,1,2,3,4,5,6,7,8,9])}
        return templates

    @staticmethod
    class ObjectView(object):
        def __init__(self, d): self.__dict__ = d

    '''Plotting Functions'''
    def plot_signals(self, 
                     xs, 
                     t, 
                     labels=None, 
                     args=None, 
                     title=None, 
                     ratio=2.6, 
                     do_transform=False, 
                     dark_mode=False, 
                     zoom=1):
        

        rows, cols = 1, 10
        fig = plt.figure(figsize=[cols*1.5,rows*1.5*ratio], dpi=60)
        for r in range(rows):
            for c in range(cols):
                ix = r*cols + c
                x, t = xs[ix], t
                
                # Ensure x is a 1D array if it's a 2D array with a single row
                if x.ndim > 1 and x.shape[0] == 1:
                    x = x.squeeze(0)
                ax = plt.subplot(rows,cols,ix+1)

                # plot the data
                if do_transform:
                        assert args is not None, "Need an args object in order to do transforms"
                        x, t = self.transform(x, t, args)  # optionally, transform the signal in some manner
                if dark_mode:
                        plt.plot(x, t, 'wo', linewidth=6)
                        ax.set_facecolor('k')
                else:
                        plt.plot(x, t, 'k-', linewidth=2)
                if labels is not None:
                        plt.title("label=" + str(labels[ix]), fontsize=22)
                plt.xlim(-zoom,zoom) ; plt.ylim(-zoom,zoom)
                plt.gca().invert_yaxis() ; plt.xticks([], []), plt.yticks([], [])
                
        if title is None:
            fig.suptitle('Noise free', fontsize=24, y=1.1)
        else:
            fig.suptitle(title, fontsize=24, y=1.1)
            
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout() ; plt.show()    

# Alternative approach: Create a modified plot_signals method
class MNIST1D_Plot_Extended(MNIST1D_Plot):
    """Extended version of MNIST1D_Plot with denoising visualization"""
    
    def plot_denoising_results(self, clean_signals, noisy_signals, denoised_signals, labels, t, 
                              save_path, title="1D Signal Denoising Results", zoom=5):
        """
        Plot denoising results using three separate calls to plot_signals logic
        """
        from utils import measure_psnr
        
        # Move to CPU and detach
        clean_signals = clean_signals.detach().cpu()
        noisy_signals = noisy_signals.detach().cpu() 
        denoised_signals = denoised_signals.detach().cpu()
        labels = labels.detach().cpu() if hasattr(labels, 'detach') else labels
        
        n_signals = min(len(clean_signals), 10)
        
        # Create combined figure
        rows, cols = 3, n_signals
        fig = plt.figure(figsize=[cols*1.5, rows*1.5*2.6], dpi=60)
        
        # Plot clean signals (top row)
        for c in range(n_signals):
            ax = plt.subplot(rows, cols, c + 1)
            x = clean_signals[c]
            if x.ndim > 1 and x.shape[0] == 1:
                x = x.squeeze(0)
            
            plt.plot(x, t, 'k-', linewidth=2)
            plt.title(f"C - Label: {int(labels[c])}", fontsize=10)
            plt.xlim(-zoom, zoom)
            plt.ylim(-zoom, zoom)
            plt.gca().invert_yaxis()
            plt.xticks([])
            plt.yticks([])
        
        # Plot noisy signals (middle row)  
        for c in range(n_signals):
            ax = plt.subplot(rows, cols, cols + c + 1)
            x = noisy_signals[c]
            if x.ndim > 1 and x.shape[0] == 1:
                x = x.squeeze(0)
                
            psnr_noisy = measure_psnr(clean_signals[c:c+1], noisy_signals[c:c+1])
            plt.plot(x, t, 'r-', linewidth=2)
            plt.title(f"N - PSNR: {psnr_noisy:.1f}dB", fontsize=10)
            plt.xlim(-zoom, zoom)
            plt.ylim(-zoom, zoom)
            plt.gca().invert_yaxis()
            plt.xticks([])
            plt.yticks([])
        
        # Plot denoised signals (bottom row)
        for c in range(n_signals):
            ax = plt.subplot(rows, cols, 2*cols + c + 1)
            x = denoised_signals[c]
            if x.ndim > 1 and x.shape[0] == 1:
                x = x.squeeze(0)
                
            psnr_denoised = measure_psnr(clean_signals[c:c+1], denoised_signals[c:c+1]).item()
            plt.plot(x, t, 'g-', linewidth=2)
            plt.title(f"D - PSNR: {psnr_denoised:.1f}dB", fontsize=10)
            plt.xlim(-zoom, zoom)
            plt.ylim(-zoom, zoom)
            plt.gca().invert_yaxis()
            plt.xticks([])
            plt.yticks([])
        
        fig.suptitle(title, fontsize=10, y=0.98)
        plt.subplots_adjust(wspace=0, hspace=0.1)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    # # Noisy Data
    noisy_dataset = MNIST1D_Dataset()
    print(noisy_dataset.data_args.iid_noise_scale, noisy_dataset.data_args.corr_noise_scale)
    noisy_data = noisy_dataset.make_dataset()


    # Clean Data 
    clean_dataset = MNIST1D_Dataset()
    clean_dataset.data_args.iid_noise_scale = 0.0
    clean_dataset.data_args.corr_noise_scale = 0.0
    print(clean_dataset.data_args.iid_noise_scale, clean_dataset.data_args.corr_noise_scale)

    clean_data = clean_dataset.make_dataset()

    Plot = MNIST1D_Plot()

    print("noisy_data['x'] shape: ", noisy_data['x'].shape)

    Plot.plot_signals(noisy_data['x'][:10], noisy_data['t'], labels=noisy_data['y'][:10], zoom = 5, title='Noise free')



    save_path = "denoising_results_extended.png"
    # Method 2: Using the extended class
    plot_extended = MNIST1D_Plot_Extended()
    plot_extended.plot_denoising_results(
        clean_data['x_test'][:10],
        noisy_data['x'][:10], 
        noisy_data['x'][:10],  # your model outputs
        noisy_data['y_test'][:10], 
        noisy_data['t'],
        save_path,
        zoom=5
    )