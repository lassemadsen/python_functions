import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import ants
import skimage.util
from pathlib import Path
from deconv_helperFunctions import mySvd, IntDcmTR, spm_nlso_gn_no_graphic, calc_CBV_by_integration
from scipy.ndimage import convolve, gaussian_filter

class DSC_process:

    def __init__(self, sub_id: str, tp: str, img_file: str, outdir:str, qc_dir: str):
        self.sub_id = sub_id
        self.tp = tp
        self.img_file = img_file
        self.qc_dir = qc_dir
        self.outdir = outdir
        self.img = nib.load(img_file)
        self.img_hdr = self.img.header
        self.img_data = self.img.get_fdata()
        self.repetition_time = self.img_hdr.get_zooms()[-1] # TR in seconds
        self.echo_time = float(str(self.img_hdr['descrip']).split(';')[0].split('TE=')[-1]) * 1e-3 # TE in seconds
        self.baseline_start = 0 # Can this be determined in a good way?

        # Parameters defined in function calls
        self.baseline_end = None
        self.conc_data = None
        self.noise_threshold = None
        self.aif = None
        self.img_data_mean = None
        self.calc_mean_image()
        self.mask = np.ones_like(self.img_data_mean).astype(bool)

        # Mask image by default? 
        #self.slice_time_correction()
        # self.mask_image()

        Path(self.qc_dir).mkdir(exist_ok=True, parents=True)
        Path(self.outdir).mkdir(exist_ok=True, parents=True)
    

    def mask_image(self, threshold: float = None):
        """ Mask image
        """
        if threshold is None:
            pct = []
            for i in [5, 15, 25, 35, 45, 55, 65]:
                pct.extend([np.percentile(self.img_data,i)])
            
            threshold_index = np.argmax(np.diff(np.diff(pct))) + 1
            threshold = pct[threshold_index]
        self.noise_threshold = threshold

        self.mask = np.any(self.img_data > threshold,axis=3)
        self.img_data[~self.mask,:] = 0

    def baseline_detection(self):
        """ 
        Performs baseline detection.
        """
        mean_signal = np.mean(self.img_data[self.mask],axis=0)

        # Smooth 
        window = 3
        moving_avg = np.convolve(mean_signal, np.ones(window)/window, mode='same')

        gradients_smooth = np.diff(moving_avg)
        threshold = -(np.max(mean_signal)-np.min(mean_signal))/20 # 5 % drop
        self.baseline_end = int(np.where(gradients_smooth < threshold)[0][0])

        self._qc_baseline_detection()

    def truncate_signal(self, bolus_length_seconds: float = 60):
        """
        Truncate signal
        """
        bolus_window = int(np.ceil(bolus_length_seconds/self.repetition_time))
        self.img_data = self.img_data[:,:,:,:self.baseline_end+1+bolus_window] # Bolus_window after last baseline, hence +1
        
        # Recalculate image mean data
        self.calc_mean_image
        
        #TODO Update header information
        self._qc_baseline_detection(truncated=True)
    
    def slice_time_correction(self, sliceorder):
        """
        Perform slice time correction. Cleaned version
        """
        if hasattr(self, 'slice_time_correction_done'):
            print('Slice time correction has already been done. Skipping...')
            return

        TimeBetweenVolumes = self.repetition_time

        nx, ny, nslices, nframes = self.img_data.shape

        # Get acqusition order
        # sliceorder = self._get_acqorder(dcm_file)
        nslices_multiband = len(np.unique(sliceorder))

        # Compute timing parameters
        TA = TimeBetweenVolumes - TimeBetweenVolumes/nslices_multiband #repetition time not same a timebetweenvolumes? 
        timing = np.array([TA / (nslices_multiband - 1), TimeBetweenVolumes - TA])
        factor = timing[0] / TimeBetweenVolumes

        # Initialize corrected data array
        slice_time_corrected = np.zeros_like(self.img_data)

        # Correct to middle of TimeBetweenVolumes
        rslice = np.floor(nslices_multiband / 2).astype(int)
        nimg = 2 ** (np.floor(np.log2(nframes)) + 1).astype(int)

        # Set up large matrix for holding image info (time by voxels)
        stack = np.zeros((nimg, nx, ny))

        for multibandii in range(nslices // nslices_multiband):
            for sliceii in range(nslices_multiband):
                
                # Set up time acquired within slice order
                shiftamount  = (np.where(sliceorder[:nslices_multiband] == sliceii)[0][0] + 1 - rslice) * factor
                
                currentslice = sliceii + multibandii * nslices_multiband

                # Set up shifting variables
                phi = np.zeros(nimg)

                # Check if signal length is odd or even
                OffSet = 0
                if nimg % 2 != 0:
                    OffSet = 1

                # Compute phase shifts
                for f in range(1,int(nimg/2)+1):
                    phi[f] = -1*shiftamount*2*np.pi/(nimg/f)

                # Mirror phi about the center
                phi[int(nimg/2-OffSet)+1:] = - np.flip(phi[1:int(nimg/2+OffSet)])

                # Compute complex exponential shifter
                shifter = (np.cos(phi) + 1j * np.sin(phi)).T
                shifter = np.tile(shifter[:, np.newaxis], (1, stack.shape[1]))  # Replicate columns

                stack[:nframes, :, :] = self.img_data[:,:,currentslice,:].T
                stack[nframes:, :, :] = np.linspace(stack[nframes - 1, :, :], stack[0, :, :], nimg - nframes, axis=0)

                # Shift the columns using FFT 
                stack = np.real(np.fft.ifft(np.fft.fft(stack, axis=0) * shifter[..., np.newaxis], axis=0))

                slice_time_corrected[:,:,currentslice, :] = stack[:nframes, :, :].T
        
        # slice_time_corrected = np.clip(slice_time_corrected, a_min=self.noise_threshold, a_max=None) # Clip to noise threshold

        # QC
        self._qc_slice_time_correction(self.img_data, slice_time_corrected)

        # Overwrite original data with slice time corrected
        self.img_data = slice_time_corrected

        # Create var to track that slice time correction has been done. 
        self.slice_time_correction_done = True
    
    def calc_mean_image(self):
        self.img_data_mean = np.mean(self.img_data,3)

    def motion_correction(self):
        """
        Motion Correction
        """
        if self.baseline_end is None:
            print('Baseline end must be set prior to motion correction.')
            return

        ants_img = ants.from_numpy(self.img_data)
        spacing = self.img_hdr.get_zooms()
        direction = np.concatenate((np.concatenate((self.img.affine[:3, :3] / spacing[:3], np.array([[0,0,0]]))), np.array([[0,0,0,1]]).T),axis=1)

        ants_img.set_spacing(spacing)
        ants_img.set_origin(list(self.img.affine[:3,3]) + [0])
        ants_img.set_direction(direction)

        fixed_img = ants_img[:,:,:,int(self.baseline_end)]

        for i in range(ants_img.shape[-1]):
            if i == self.baseline_end:
                continue
            moving_img = ants_img[:,:,:,i]
            mytx = ants.registration(fixed_img, moving_img, type_of_transform='DenseRigid')
            ants_img[:,:,:,i] = ants.apply_transforms(fixed=fixed_img, moving=moving_img, transformlist=mytx['fwdtransforms'])

        self.img_data = ants_img.numpy()

        #TODO Check overwrite image data 
        #TODO Recalcualte mean image data
        self.calc_mean_image()
        self._qc_motion_correction()

    def calc_concentration(self, k: float = 1):
        """
        Calculation of contrast agent concentration

        C(t) ∝ ΔR2 = -k/TE*ln(S(t)/S0)

        Parameters
        ----------
        k : float | 1
            Proportionality factor
        """
        self.conc_data = np.zeros_like(self.img_data)

        intensity_data = self.img_data[self.mask]
        S0 = np.mean(intensity_data[:,self.baseline_start:self.baseline_end+1],axis=1)

        # Deal with values going to zero:
        too_small_peak = np.where(intensity_data <= 0)
        for idx, t in zip(*too_small_peak):
            std = np.std(intensity_data[idx][self.baseline_start:self.baseline_end+1], ddof=1)
            if np.any(intensity_data[idx] > 0):
                floor_value = np.min([np.min(intensity_data[idx][intensity_data[idx] > 0]), std])
            else:
                floor_value = std
            intensity_data[idx,t] = floor_value

        # Calculate concentration
        self.conc_data[self.mask] = -k / self.echo_time * np.log(intensity_data / S0[...,np.newaxis])

        # Check for NaNs or Infs
        self.conc_data[np.isnan(self.conc_data)] = 0
        self.conc_data[np.isinf(self.conc_data)] = 0

        # TODO Option to mask 5 and 95 percentiles of CBV? 

        self._qc_concentration()

    def aif_selection(self, aif_search_mask: str, gm_mask: str, n_aif: int = 10):
        aif_search_mask = nib.load(aif_search_mask)
        gm_mask = nib.load(gm_mask)

        # TODO Check header information between aif_search_mask, gm_mask and self.img fits (i.e. dimensions, steps, location etc.)

        from aif_selection import aif_selection
        self.aif_select = aif_selection(self, aif_search_mask.get_fdata(), gm_mask.get_fdata(), n_aif)
        self.aif_select.select_aif()
        self.aif = self.aif_select.final_aif
        self.aif_area = np.trapz(self.aif, self.repetition_time)

        self._qc_aif_selection()

    def calc_perfusion(self, sampling_factor:int = 8, TimeBetweenVolumes:float = None):
        #TODO Calc TTP 
        if TimeBetweenVolumes is None:
            TimeBetweenVolumes = self.repetition_time

        # Initialize parameter images:
        self.alpha_img = np.zeros_like(self.img_data_mean)
        self.beta_img = np.zeros_like(self.img_data_mean)
        self.delay_img = np.zeros_like(self.img_data_mean)
        self.cbf_img = np.zeros_like(self.img_data_mean)
        self.cbv_img = np.zeros_like(self.img_data_mean)
        self.mtt_img = np.zeros_like(self.img_data_mean)
        self.cth_img = np.zeros_like(self.img_data_mean)
        self.rth_img = np.zeros_like(self.img_data_mean)

        t = np.arange(self.conc_data.shape[3]) * TimeBetweenVolumes # OBS: Should probably be repetition time, however this is different from TimeBetweenVolumes in MATLAB 1.56 vs. 1.563

        from tqdm import tqdm
        for z in tqdm(range(self.conc_data.shape[2]), desc="Slice"):
            if not np.any(self.mask[:, :, z]):
                continue # No valid data in slice
            # Slice-wise initial guess:
            # Find valid voxels
            voxels = np.argwhere(self.mask[:, :, z])
            voxels = voxels[np.lexsort((voxels[:, 0], voxels[:, 1]))] # Sort same as matlab

            svd_cbf, svd_delay, svd_cbv, svd_rf  = mySvd(self.conc_data[voxels[:,0], voxels[:,1], z, :], self.aif, self.baseline_end, TimeBetweenVolumes)

            cbvbyC = np.array([calc_CBV_by_integration(self.conc_data[voxels[i,0], voxels[i,1], z, self.baseline_end:], TimeBetweenVolumes, self.aif_area) for i in range(len(voxels))])

            svd_mtt = np.array([cbvbyC[i]/svd_cbf[i] if svd_cbf[i] != 0 else 0 for i in range(len(voxels))])

            # Adjust initial paramters if they are beyond the limits
            svd_delay[svd_delay == 0] = 1 # 1 sec or TimeBetweenVolumes/sampling_factor # Delay of 0 Will cause problems when log transforming paramters for optimization (log(0) = -Inf)

            # For MATLAB compatability. A bit messy to upsample here. Should be done in fitting class
            dt = 1 / sampling_factor * TimeBetweenVolumes 
            svd_delay = svd_delay / dt
            # Control minimum delay (only above this in very noisy voxels.)
            svd_delay = np.minimum(svd_delay, 5/dt)

            svd_mtt[svd_mtt <= 0] = 1
            svd_cbf[svd_cbf == 0] = np.min(svd_cbf[svd_cbf != 0])

            p_slice = [np.log(np.array([svd_cbf[i], 1, svd_delay[i], svd_mtt[i]])) for i in range(self.mask[:, :, z].sum())]

            for i in tqdm(range(len(voxels)), desc='Voxel', leave=False):
                x = voxels[i, 0]
                y = voxels[i, 1]
                voxel_data = self.conc_data[x,y,z]

                cbv, cbf, alpha, beta, delay, mtt, cth, rth = self._calc_perfusion_voxel(voxel_data, t, p_slice[i], sampling_factor, TimeBetweenVolumes, show_fitting_progression=False)

                self.alpha_img[x,y,z] = alpha
                self.beta_img[x,y,z] = beta
                self.delay_img[x,y,z] = delay
                self.cbf_img[x,y,z] = cbf
                self.cbv_img[x,y,z] = cbv
                self.mtt_img[x,y,z] = mtt
                self.cth_img[x,y,z] = cth
                self.rth_img[x,y,z] = rth

        # Save parametric images
        self._save_img(self.alpha_img, 'ALPHA')
        self._save_img(self.beta_img, 'BETA')
        self._save_img(self.delay_img, 'DELAY')
        self._save_img(self.cbf_img, 'CBF')
        self._save_img(self.cbv_img, 'CBV')
        self._save_img(self.mtt_img, 'MTT')
        self._save_img(self.cth_img, 'CTH')
        self._save_img(self.rth_img, 'RTH')

    def smooth_data(self, smooth_mask: np.array):
        # Slice-wise smoothing of image
        # TODO Options (gaussian/uniform, filter size)

        # conc_data_smoothed = np.zeros_like(self.conc_data)
        fwhm = 1.5 # From matlab. Not sure why this is selected. Half a voxel? Should be based on voxel size
        # sigma = fwhm / np.sqrt(8 * np.log(2))
        # kernel = gaussian_filter((np.arange(9) == 4).reshape(3,3).astype(float), sigma)
        kernel = self._get_kernel(fwhm, 3)
        edge_width = int((kernel.shape[0]-1)/2)

        data_smoothed = np.zeros_like(self.conc_data)

        smooth_mask = smooth_mask * self.mask

        for frame in range(self.conc_data.shape[-1]):
            for z_slice in range(5,self.conc_data.shape[2]):
                slice_data = self.conc_data[:,:,z_slice,frame]

                # Step 1: Smooth all voxels with valid conc data
                slice_mask = (self.mask[:,:,z_slice] != 0).astype(float)
                smoothed_slice = convolve(slice_data * slice_mask, kernel, mode='constant', cval=0)

                # Scale edge voxels with valid voxels within kernel
                count_valid = convolve(slice_mask, kernel, mode='constant', cval=0) * slice_mask
                smoothed_slice = np.divide(smoothed_slice, count_valid, out=np.zeros_like(smoothed_slice), where=count_valid != 0)
                self._zero_edge(smoothed_slice, edge_width)
                data_smoothed[:,:,z_slice, frame] = smoothed_slice

                if smooth_mask is not None:
                    # Step 2: Smooth all voxels and only keep non-zero voxels from original data
                    slice_smoothing_mask = (smooth_mask[:,:,z_slice] != 0).astype(float)
                    smoothed_masked_slice = convolve(slice_data * slice_smoothing_mask, kernel, mode='constant', cval=0) 
                    # Scale edge voxels with valid voxels within kernel
                    count_valid = convolve(slice_smoothing_mask, kernel, mode='constant', cval=0) * slice_smoothing_mask
                    smoothed_masked_slice = np.divide(smoothed_masked_slice, count_valid, out=np.zeros_like(smoothed_masked_slice), where=count_valid != 0)

                    data_smoothed[:,:,z_slice,frame][smooth_mask[:,:,z_slice] != 0] = smoothed_masked_slice[smooth_mask[:,:,z_slice] != 0]

                # Set edges to zero
                data_smoothed[:,:,z_slice,frame] = self._zero_edge(data_smoothed[:,:,z_slice,frame], edge_width)
                self.mask[:,:,z_slice] = self._zero_edge(self.mask[:,:,z_slice], edge_width)
            
        self.conc_data = data_smoothed

    def _get_kernel(self, fwhm, kernel_width):
        """ Funciton to get smoothing kernel exactly like DSC-pipeline in MATLAB (CFIN repo: gauss_kern.m)
        """
        sigma = fwhm / np.sqrt(8 * np.log(2)) + np.finfo(float).eps

        lx = (kernel_width-1)/2
        Ex = np.min([np.ceil(3*sigma), lx])
        x = np.arange(-Ex, Ex+Ex/2) # +Ex/2 to make sure last point is included
        kx = np.exp(-x**2 / (2 * sigma**2))
        kx = kx/np.sum(kx)
        kernel = np.tile(kx, (kernel_width,1)) * np.tile(kx, (kernel_width,1)).T

        return kernel

    def _zero_edge(self, img_data: np.array, edge_width: int):

        if len(img_data.shape) == 3: # If 3D image
            for z in range(img_data.shape[2]):
                img_data[:edge_width,:,z] = 0       # Top edge
                img_data[-edge_width:,:,z] = 0      # Bottom edge
                img_data[:, :edge_width,z] = 0      # Left edge
                img_data[:, -edge_width:,z] = 0     # Right edge
        elif len(img_data.shape) == 2: #If 2D image
                img_data[:edge_width,] = 0          # Top edge
                img_data[-edge_width:,] = 0         # Bottom edge
                img_data[:, :edge_width] = 0        # Left edge
                img_data[:, -edge_width:] = 0       # Right edge
        else:
            print('Error')
            return

        return img_data

    def _calc_perfusion_voxel(self, y: np.array, t: np.array, p: np.array, sampling_factor: int, TimeBetweenVolumes: float, show_fitting_progression=False):
        # Only use bolus passage in the remaining optimization
        y = y[self.baseline_end:]
        t_bolus = t[self.baseline_end:] - t[self.baseline_end] # Time vector from baseline end

        # Initialize class for fitting the parametric function 
        int_dcmTR = IntDcmTR(self.aif[self.baseline_end:], sampling_factor, TimeBetweenVolumes, save_progression=show_fitting_progression)

        # Setup priors
        pC = np.diag([.1, 1, 10, .1]) 

        # Run optimization algorithm 
        n_iterations = 2
        rmse = {}
        estimated_parameters = {}
        fitted_values = {}
        selected_iteration = None

        for iteration in range(n_iterations):
            Ep, Cp, S, F = spm_nlso_gn_no_graphic(int_dcmTR.fit, p, pC, y, t_bolus)
            fitted_values[iteration] = int_dcmTR.fit(t_bolus, Ep)

            sumsq = np.sum(np.power(y-fitted_values[iteration],2))
            rmse[iteration] = np.sqrt(sumsq)/np.sum(np.abs(y))

            if iteration == 0:
                # Update initial guessing paramters and see if this improves the fitting.
                p[0] = Ep[0]
                p[1] = 0
                p[2] = Ep[2]
                p[3] = Ep[1] + Ep[3]
                selected_iteration = iteration
            else:
                if rmse[iteration] > 0.01 and not (rmse[iteration] > rmse[iteration - 1]) and (np.abs(rmse[iteration] - rmse[iteration -1]) / rmse[iteration -1]) > 0.10:
                    selected_iteration = iteration
                elif (np.abs(rmse[iteration] - rmse[iteration -1]) / rmse[iteration -1] <= 0.10) or rmse[iteration] > rmse[iteration - 1]:
                    # No major improvement
                    selected_iteration = iteration - 1

            estimated_parameters[iteration] = Ep

        if show_fitting_progression:
            for i in range(len(int_dcmTR.y_progression)):
                plt.cla()
                plt.plot(int_dcmTR.y_progression[i, :], label=f'Fit {i}', color='orange')
                plt.title(f'Progression Step {i+1}')
                plt.legend()
                plt.scatter(range(len(y)), y, label='Target Data')  # Replot scatter after clearing
                plt.plot(y, label='Target Data')
                plt.pause(0.01)

        # Calculate derived parameters
        cbv = np.trapz(fitted_values[selected_iteration]) * TimeBetweenVolumes/self.aif_area
        cbf = np.exp(estimated_parameters[selected_iteration][0]) # Should be multiplied by conc_area to normalize
        alpha = np.exp(estimated_parameters[selected_iteration][1])
        beta = np.exp(estimated_parameters[selected_iteration][3])
        delay = np.exp(estimated_parameters[selected_iteration][2])
        mtt = alpha * beta
        cth = np.sqrt(alpha) * beta #(alpha**beta) * beta
        rth = cth/mtt
        # TODO limits on MTT and CTH?

        return cbv, cbf, alpha, beta, delay, mtt, cth, rth


    # SET methods
    def set_baseline_end(self, baseline_end: int):
        self.baseline_end = baseline_end
    
    def set_baseline_start(self, baseline_start: int):
        self.baseline_start = baseline_start

    def set_aif_search_mask(self, aif_seach_mask: str):
        mask = nib.load(aif_seach_mask)
        mask_hdr = mask.header
        # TODO Check header fits with DSC data

        self.aif_seach_mask = mask.get_fdata()

    # QC methods
    def _qc_baseline_detection(self, truncated: bool = False):
        mean_signal = np.mean(self.img_data[self.mask],axis=0)
        plt.figure()
        plt.plot(range(len(mean_signal))*self.repetition_time, mean_signal)
        plt.scatter(self.baseline_end*self.repetition_time, mean_signal[self.baseline_end])
        plt.ylabel('Average signal intensity')
        plt.xlabel('Time (s)')

        if truncated:
            plt.title(f'Bolus truncation: {self.sub_id} - {self.tp}')
            plt.savefig(f'{self.qc_dir}/{self.sub_id}_{self.tp}_bolus_truncation.jpg', dpi=200)
        else:
            plt.title(f'Baseline detection: {self.sub_id} - {self.tp}')
            plt.savefig(f'{self.qc_dir}/{self.sub_id}_{self.tp}_baseline_detection.jpg', dpi=200)

        plt.close()
    
    def _qc_motion_correction(self, n_cols: int = 8, slice_number: int = 25):
        #TODO maybe plot pre and post motion correction 
        if self.baseline_end is None:
            print('Baseline end must be set.')
            return

        dif_images = np.stack([self.img_data[:,:,:,i] - self.img_data[:,:,:,self.baseline_end] for i in range(self.img_data.shape[3])], axis=-1)

        m = skimage.util.montage([dif_images[:,:,slice_number,i] for i in range(dif_images.shape[3])], grid_shape=(np.ceil(dif_images.shape[3]/n_cols), n_cols))
        
        plt.figure()
        plt.imshow(m, cmap='gray')
        plt.axis('off')
        plt.title(f'Motion correction: {self.sub_id} - {self.tp}')
        plt.savefig(f'{self.qc_dir}/{self.sub_id}_{self.tp}_motion_correction.jpg', dpi=200)
        plt.close()
                    
    def _qc_concentration(self):
        mean_conc = np.mean(self.conc_data[self.mask],axis=0)
        plt.figure()
        plt.plot(range(len(mean_conc))*self.repetition_time, mean_conc)
        plt.scatter(self.baseline_end*self.repetition_time, mean_conc[self.baseline_end])
        plt.title(f'Average concentration curve: {self.sub_id} - {self.tp}')
        plt.ylabel('Concentration')
        plt.xlabel('Time (s)')
        plt.savefig(f'{self.qc_dir}/{self.sub_id}_{self.tp}_concentration.jpg', dpi=200)
        plt.close()

    def _qc_aif_selection(self):
        # Check that aif_selection has been run
        if self.aif is None:
            print('AIF selection has not been performed.')
        else:
            self.aif_select.qc_aif(f'{self.qc_dir}/{self.sub_id}_{self.tp}_aif_selection.jpg')

    def _qc_slice_time_correction(self, original: np.array, slice_time_corrected: np.array):
        # QC: Compute mean along first two axes and find min position
        fig, ax = plt.subplots(1,2, figsize=(14,6))
        n_slices = self.img_data.shape[2]
        minpos = np.zeros(n_slices)
        minpostps = np.zeros(n_slices)

        for i in range(n_slices):
            avg_signal_pre_correction = np.mean(original[self.mask[:,:,0],i,:],axis=0)
            minpos[i] = np.argmin(avg_signal_pre_correction)*self.repetition_time

            avg_signal_post_correction = np.mean(slice_time_corrected[self.mask[:,:,0],i,:],axis=0)
            minpostps[i] = np.argmin(avg_signal_post_correction)*self.repetition_time

            if i == 0: # Only label on fist iteration
                label_orig = 'Original'
                label_corrected = 'Slice time corrected'
            else:
                label_orig = None
                label_corrected = None

            ax[0].plot(self.repetition_time*range(len(avg_signal_pre_correction)), avg_signal_pre_correction, 'r', label=label_orig, alpha=0.8)
            ax[0].plot(self.repetition_time*range(len(avg_signal_post_correction)), avg_signal_post_correction, 'b', label=label_corrected, alpha=0.8)
            

        ax[0].set_title('Average slice curves')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Signal intensity')
        ax[0].legend(loc='upper right')

        ax[1].plot(minpos, 'b-*', label='Original')
        ax[1].plot(minpostps, 'ro--', label='Slice time corrected')
        ax[1].set_xlabel('Slice')
        ax[1].set_ylabel('Time to minimum (s)')
        ax[1].set_title('Time to minimum of slice average curve')
        ax[1].legend(loc='upper left')

        fig.suptitle(f'Slice time correction {self.sub_id}, {self.tp}')

        plt.savefig(f'{self.qc_dir}/{self.sub_id}_{self.tp}_slice_time_correction.jpg', dpi=200)
        plt.close()


    # Save function
    def _save_img(self, data: np.array, name: str):
        #TODO Maybe track progression of analysis
        #TODO should header be different? 

        outdir = f'{self.outdir}/{name}/NATPACE'
        Path(outdir).mkdir(exist_ok=True, parents=True)

        img_to_save = nib.Nifti1Image(data, self.img.affine, self.img_hdr)
        nib.save(img_to_save, outdir + '/0001.nii')

    def _get_acqorder(self, dcm_file):
        import pydicom
        ds = pydicom.dcmread(dcm_file)
        for element in ds:
            if "MosaicRefAcqTimes" in element.name:
                acqtime = element.value

        acqorder = np.argsort(acqtime)

        uniq_acqtime = list(dict.fromkeys(acqtime))

        _, uniq_acqorder = np.unique(uniq_acqtime, return_index=True)

        for i, time in enumerate(uniq_acqtime):
            acqorder[np.array(acqtime) == time] = uniq_acqorder[i]
        
        return acqorder