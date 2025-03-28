import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import ants
import skimage.util
from pathlib import Path
from deconv_helperFunctions import mySvd, IntDcmTR, spm_nlso_gn_no_graphic
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
        self.img_data_mean = np.mean(self.img_data,3)
        self.repetition_time = self.img_hdr.get_zooms()[-1] # TR in seconds
        self.echo_time = float(str(self.img_hdr['descrip']).split(';')[0].split('TE=')[-1]) * 1e-3 # TE in seconds
        self.baseline_start = 0 # Can this be determined in a good way?

        # Parameters defined in function calls
        self.baseline_end = None
        self.conc_data = None
        self.noise_threshold = None
        self.aif = None
        self.mask = None # TODO Create mask 

        # Mask image by default 
        #self.slice_time_correction()
        self.mask_image()

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

        self.mask = self.img_data > threshold
        self.img_data[self.img_data <= threshold] = np.nan
        self.img_data[np.any(np.isnan(self.img_data),axis=3)] = np.nan

    def baseline_detection(self):
        """ 
        Performs baseline detection.
        """
        mean_signal = np.nanmean(self.img_data, (0,1,2))

        # Smooth 
        window = 3
        moving_avg = np.convolve(mean_signal, np.ones(window)/window, mode='same')

        gradients_smooth = np.diff(moving_avg)
        threshold = -(np.nanmax(mean_signal)-np.nanmin(mean_signal))/20 # 5 % drop
        self.baseline_end = int(np.where(gradients_smooth < threshold)[0][0])

        self._qc_baseline_detection()

    def truncate_signal(self, bolus_length_seconds: float = 60):
        """
        Truncate signal
        """
        bolus_window = int(np.ceil(bolus_length_seconds/self.repetition_time))
        self.img_data = self.img_data[:,:,:,:self.baseline_end+1+bolus_window] # Bolus_window after last baseline, hence +1
        
        # Recalculate image mean data
        self.img_data_mean = np.mean(self.img_data,3)
        
        #TODO Update header information
        self._qc_baseline_detection(truncated=True)
    
    def slice_time_correction(self, dcm_file):
        """
        Perform slice time correction. Cleaned version
        """
        if hasattr(self, 'slice_time_correction_done'):
            print('Slice time correction has already been done. Skipping...')
            return

        TimeBetweenVolumes = 1.56

        nx, ny, nslices, nframes = self.img_data.shape

        # Get acqusition order
        sliceorder = self._get_acqorder(dcm_file)
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
        
        slice_time_corrected = np.clip(slice_time_corrected, a_min=self.noise_threshold, a_max=None) # Clip to noise threshold

        # QC
        self._qc_slice_time_correction(self.img_data, slice_time_corrected)

        # Overwrite original data with slice time corrected
        self.img_data = slice_time_corrected

        # Create var to track that slice time correction has been done. 
        self.slice_time_correction_done = True

    def motion_correction(self):
        """
        Motion Correction
        """
        if self.baseline_end is None:
            print('Baseline end must be set prior to motion correction.')
            return

        # Convert NaN to 0 for ANTs registration to work 
        ants_img = ants.from_numpy(np.nan_to_num(self.img_data)) 
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

        # Convert back to NaN
        ants_img_data = ants_img.numpy()
        ants_img_data[np.isnan(self.img_data)] = np.nan
        self.img_data = ants_img_data

        #TODO Check overwrite image data 
        #TODO Recalcualte mean image data
        self._qc_motion_correction()

    def concentration_calc(self, k: float = 1):
        """
        Calculation of contrast agent concentration

        C(t) ∝ ΔR2 = -k/TE*ln(S(t)/S0)

        Parameters
        ----------
        k : float | 1
            Proportionality factor
        """
        S0 = np.mean(self.img_data[:,:,:,self.baseline_start:self.baseline_end+1],axis=3) # Including last baseline point
        self.conc_data = -k/self.echo_time*np.log(self.img_data/S0[..., np.newaxis])

        self._qc_concentration()

    def aif_selection(self, aif_search_mask: str, gm_mask: str, n_aif: int = 10):
        aif_search_mask = nib.load(aif_search_mask)
        gm_mask = nib.load(gm_mask)

        # TODO Check header information between aif_search_mask, gm_mask and self.img fits (i.e. dimensions, steps, location etc.)

        from aif_selection import aif_selection
        self.aif_select = aif_selection(self, aif_search_mask.get_fdata(), gm_mask.get_fdata(), n_aif)
        self.aif_select.select_aif()
        self.aif = self.aif_select.final_aif

        self._qc_aif_selection()

    def calc_perfusion(self, smooth_mask:str = None, sampling_factor:int = 8):
        # Smoothing mask
        self.mask = ~np.all(self.conc_data == 0, axis=-1) # TODO Only for debugging
        self._smooth_data(smooth_mask)

        # Calc TTP 
        aif_area = np.trapz(self.aif) * self.repetition_time # In MATLAB, the aif_area is read from the AIF info file. This is a bit different (not sure why) and may cause tiny differences in the results compared to MATLAB.
        TimeBetweenVolumes = 1.56 #self.repetition_time

        # Initialize parameter images:
        alpha_img = np.zeros_like(self.img_data_mean)
        beta_img = np.zeros_like(self.img_data_mean)
        delay_img = np.zeros_like(self.img_data_mean)
        cbf_img = np.zeros_like(self.img_data_mean)
        cbv_img = np.zeros_like(self.img_data_mean)
        mtt_img = np.zeros_like(self.img_data_mean)
        cth_img = np.zeros_like(self.img_data_mean)
        rth_img = np.zeros_like(self.img_data_mean)

        # idx = np.unravel_index(911, self.conc_data.shape, order='F')
        # conc_voxel = self.conc_data[idx[0],idx[1],idx[2],:] # First voxel in mask slice 0 matlab

        t = np.arange(self.conc_data.shape[3]) * TimeBetweenVolumes # OBS: Should probably be repetition time, however this is different from TimeBetweenVolumes in MATLAB 1.56 vs. 1.563

        from tqdm import tqdm
        for z in tqdm(range(self.conc_data.shape[2]), desc="Slice"):
            # Slice wise initial guess:
            # Compute SVD only where mask is True and store results #TODO AIF matrix in mySvd does not have to be calculated each time. Could be more made more efficent, however it is still rather fast.     
            svd_res = [mySvd(self.conc_data[x, y, z, :], self.aif, self.baseline_end, TimeBetweenVolumes) for x in range(self.conc_data.shape[0]) for y in range(self.conc_data.shape[1]) if self.mask[x, y, z]]
            svd_cbf, svd_delay, _, _ = map(np.array,zip(*svd_res))

            cbvbyC = np.array([np.trapz(np.clip(self.conc_data[x, y, z, :], a_min=0, a_max=None), dx=TimeBetweenVolumes)/aif_area for x in range(self.conc_data.shape[0]) for y in range(self.conc_data.shape[1]) if self.mask[x, y, z]])
            svd_mtt = cbvbyC/svd_cbf

            # Adjust initial paramters if they are beyond the limits
            svd_delay[svd_delay == 0] = TimeBetweenVolumes/sampling_factor # Delay of 0 Will cause problems when log transforming paramters for optimization (log(0) = -Inf)
            svd_mtt[svd_mtt <= 0] = 1
            svd_cbf[svd_cbf <= 0] = np.min(svd_cbf[svd_cbf >= 0])

            p_slice = [np.log(np.array([svd_cbf[i], 1, svd_delay[i], svd_mtt[i]])) for i in range(self.mask[:, :, z].sum())]

            # Perform voxel-wise deconvolution
            for mask_index, (x, y) in tqdm(enumerate(np.argwhere(self.mask[:,:,z])), total=self.mask[:, :, z].sum(), desc='Voxel', leave=False): # Keep track of index according to p_slice
            # for mask_index, (x, y) in enumerate(np.argwhere(mask[:,:,z])): # Keep track of index according to p_slice
                voxel_data = self.conc_data[x,y,z]

                cbv, cbf, alpha, beta, delay, mtt, cth, rth = self._calc_perfusion_voxel(voxel_data, t, p_slice[mask_index], sampling_factor, TimeBetweenVolumes)

                alpha_img[x,y,z] = alpha
                beta_img[x,y,z] = beta
                delay_img[x,y,z] = delay
                cbf_img[x,y,z] = cbf
                cbv_img[x,y,z] = cbv
                mtt_img[x,y,z] = mtt
                cth_img[x,y,z] = cth
                rth_img[x,y,z] = rth

        # Save parametric images
        self._save_img(alpha_img, 'ALPHA')
        self._save_img(beta_img, 'BETA')
        self._save_img(delay_img, 'DELAY')
        self._save_img(cbf_img, 'CBF')
        self._save_img(cbv_img, 'CBV')
        self._save_img(mtt_img, 'MTT')
        self._save_img(cth_img, 'CTH')
        self._save_img(rth_img, 'RTH')

    def _smooth_data(self, smooth_mask):
        # Slice-wise smoothing of image
        # TODO Options (gaussian/uniform, filter size)

        conc_data_smoothed = np.zeros_like(self.conc_data)
        fwhm = 1.5 # From matlab. Not sure why this is selected. Half a voxel? Should be based on voxel size
        sigma = fwhm / np.sqrt(8 * np.log(2))
        kernel = gaussian_filter((np.arange(9) == 4).reshape(3,3).astype(float), sigma)
        # baseline_start = 4 # TODO is this nesesary? Perhaps just smooth from bl_end. In that case 

        if smooth_mask is not None:
            smooth_mask = nib.load(smooth_mask).get_fdata() # TODO Check header. 
            smooth_mask = smooth_mask * self.mask
        else:
            smooth_mask = None

        for frame in range(self.conc_data.shape[-1]):
            for z_slice in range(self.conc_data.shape[2]):
                slice_data = self.conc_data[:,:,z_slice,frame]
                # Step 1: Smooth all voxels and only keep non-zero voxels from original data
                slice_mask = (self.mask[:,:,z_slice] != 0).astype(float)
                smoothed_slice = convolve(slice_data, kernel, mode='constant', cval=0)

                # Scale edge voxels with valid voxels within kernel
                count_valid = convolve(slice_mask, kernel, mode='constant', cval=0)
                smoothed_slice = np.divide(smoothed_slice, count_valid, out=np.zeros_like(smoothed_slice), where=count_valid != 0)
                # #TODO remove edge voxels 

                conc_data_smoothed[:,:,z_slice] = smoothed_slice

                if smooth_mask is not None:
                    slice_smoothing_mask = (smooth_mask[:,:,z_slice] != 0).astype(float)
                    smoothed_masked_slice = convolve(slice_data, kernel, mode='constant', cval=0)
                    # Scale edge voxels with valid voxels within kernel
                    count_valid = convolve(slice_smoothing_mask, kernel, mode='constant', cval=0)
                    smoothed_masked_slice = np.divide(smoothed_masked_slice, count_valid, out=np.zeros_like(smoothed_masked_slice), where=count_valid != 0)

                    # Overwrite voxels in smooth_mask
                    conc_data_smoothed[smooth_mask[:,:,z_slice] != 0] = smoothed_masked_slice[smooth_mask[:,:,z_slice] != 0]

        return conc_data_smoothed

    def _calc_perfusion_voxel(self, y, t, p, sampling_factor, TimeBetweenVolumes):
        # Only use bolus passage in the remaining optimization
        y = y[self.baseline_end:]
        t_bolus = t[self.baseline_end:] - t[self.baseline_end] # Time vector from baseline end

        # Initialize class for fitting the parametric function 
        int_dcmTR = IntDcmTR(self.aif[self.baseline_end:], sampling_factor, TimeBetweenVolumes) # Could this be move out? 

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

        # Calculate derived parameters
        cbv = np.trapz(fitted_values[selected_iteration])
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
    def _qc_baseline_detection(self, truncated=False):
        mean_signal = np.nanmean(self.img_data, (0,1,2))
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
        np.nan_to_num(m, 0)
        
        plt.figure()
        plt.imshow(m, cmap='gray')
        plt.axis('off')
        plt.title(f'Motion correction: {self.sub_id} - {self.tp}')
        plt.savefig(f'{self.qc_dir}/{self.sub_id}_{self.tp}_motion_correction.jpg', dpi=200)
        plt.close()
                    
    def _qc_concentration(self):
        mean_conc = np.nanmean(self.conc_data, (0,1,2))
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

    def _qc_slice_time_correction(self, original, slice_time_corrected):
        # QC: Compute mean along first two axes and find min position
        fig, ax = plt.subplots(1,2, figsize=(14,6))
        n_slices = self.img_data.shape[2]
        minpos = np.zeros(n_slices)
        minpostps = np.zeros(n_slices)

        for i in range(n_slices):
            # avg_signal_pre_correction = np.nanmean(original[:,:,i,:], (0,1))
            avg_signal_pre_correction = np.mean(np.nan_to_num(original[:,:,i,:]), (0,1))
            minpos[i] = np.nanargmin(avg_signal_pre_correction)*self.repetition_time

            # avg_signal_post_correction = np.nanmean(slice_time_corrected[:,:,i,:], (0,1))
            avg_signal_post_correction = np.mean(np.nan_to_num(slice_time_corrected[:,:,i,:]), (0,1))
            minpostps[i] = np.nanargmin(avg_signal_post_correction)*self.repetition_time

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
    def _save_img(self, data, name):
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