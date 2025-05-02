import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("Agg") # Plot in background
from matplotlib.gridspec import GridSpec
from DSC_processing import DSC_process

class aif_selection:

    def __init__(self, dsc_process: DSC_process, aif_search_mask: np.array, gm_mask: np.array, n_aif: int):
        self.dsc_process = dsc_process
        self.n_aif = n_aif
        self.aif_search_mask = aif_search_mask
        self.gm_mask = gm_mask

    def select_aif(self):

        # Find AIF search voxels 
        self.aifSearch_indices = np.where((self.aif_search_mask == 1) & (np.all(~np.isnan(self.dsc_process.img_data),axis=3)))

        self.aifSearchVoxels_conc = self.dsc_process.conc_data[self.aifSearch_indices[0], self.aifSearch_indices[1], self.aifSearch_indices[2], :]
        self.aifSearchVoxels_signal = self.dsc_process.img_data[self.aifSearch_indices[0], self.aifSearch_indices[1], self.aifSearch_indices[2], :]

        # Get mean concentration curve in gray matter
        gm_indices = np.where(self.gm_mask)
        self.mean_gm_conc = np.nanmean(self.dsc_process.conc_data[gm_indices[0], gm_indices[1], gm_indices[2], :], axis=0)
        self.mean_gm_signal = np.nanmean(self.dsc_process.img_data[gm_indices[0], gm_indices[1], gm_indices[2], :], axis=0)
        gm_peak_idx = np.argmax(self.mean_gm_conc)
        sec_pass_gm = np.argmax(self.mean_gm_conc) + np.where(np.diff(self.mean_gm_conc[np.argmax(self.mean_gm_conc):]) > 0)[0][0] # Index of minimum before second pass

        # Calculate AIF metrics in each AIF serach voxel
        aif_metrics = [self.get_AIF_metrics(self.aifSearchVoxels_conc[i], gm_peak_idx, sec_pass_gm) for i in range(len(self.aifSearchVoxels_conc))]
        auc, peak, fwhm, baseline_roughness, up_area, TTP, gamma_rmse, M = zip(*aif_metrics)
        self.df_aif_metrics = pd.DataFrame(np.array([auc, peak, fwhm, baseline_roughness, up_area, TTP, gamma_rmse, M]).T, 
                                      columns=['auc', 'peak', 'fwhm', 'bl_roughness', 'up_area', 'TTP', 'gamma_rmse', 'M'])

        # Exclude voxels with NaN in AIFmetrics and AUC < 0
        self.df_aif_metrics = self.df_aif_metrics.dropna()
        self.df_aif_metrics = self.df_aif_metrics[self.df_aif_metrics.auc > 0]

        # Estimate noise level in both signal (4 * std) and concentration (3 * std) curves. The noise is esitamted at baseline (not including last baseline point).
        self.noise_level_signal = np.mean(np.std(self.aifSearchVoxels_signal[:,self.dsc_process.baseline_start:self.dsc_process.baseline_end-1],axis=1)) * 4 # Do not include last two baseline points as an AIF might rise early
        self.noise_level_conc = - np.mean(np.std(self.aifSearchVoxels_conc[:,self.dsc_process.baseline_start:self.dsc_process.baseline_end-1],axis=1)) * 3 # Do not include last two baseline points as an AIF might rise early

        # Exclude curve with signal below noise level
        self.df_aif_metrics = self.df_aif_metrics.iloc[np.where(~(self.aifSearchVoxels_signal[self.df_aif_metrics.index] < self.noise_level_signal).any(axis=1))[0]]
        self.df_aif_metrics = self.df_aif_metrics.iloc[np.where(~(self.aifSearchVoxels_conc[self.df_aif_metrics.index] < self.noise_level_conc).any(axis=1))[0]]

        # Perform AIF selection based on the criteria below
        criteria = [
            {'column': 'auc', 'percentile': 50, 'operation': '>'},
            {'column': 'bl_roughness', 'percentile': 90, 'operation': '<'},
            {'column': 'TTP', 'percentile': 50, 'operation': '<'},
            {'column': 'gamma_rmse', 'percentile': 75, 'operation': '<'},
            {'column': 'fwhm', 'percentile': 50, 'operation': '<'},
            {'column': 'up_area', 'percentile': 50, 'operation': '>'}
        ]

        # Get the indicies of AIFs
        self.aif_indices = self.filter_indices(self.df_aif_metrics, criteria)

        if len(self.aif_indices) < self.n_aif:
            if len(self.aif_indices) == 0:
                print(f'Error! No AIF candidates found.. ')
                return
            else:
                print(f'Warning! Number of AIF candidates ({len(self.aif_indices)}) less then selected number AIFs ({self.n_aif})')
                self.n_aif = len(self.aif_indices)
        self.final_aif_indices = self.df_aif_metrics.iloc[self.aif_indices].sort_values(by='M', ascending=False).index[:self.n_aif]

        # Calculate final AIF as the median of the n_aif (e.g. 10) best AIF curves (based on M)
        self.final_aifs = self.aifSearchVoxels_conc[self.final_aif_indices]
        self.final_aif = np.mean(self.final_aifs, axis=0)
        self.final_aif_signal = np.mean(self.aifSearchVoxels_signal[self.final_aif_indices], axis=0)

    def get_AIF_metrics(self, voxel_signal, gm_peak_idx, sec_pass_gm):
        # Find time-to-peak
        TTP = np.argmax(voxel_signal)

        if TTP <= self.dsc_process.baseline_end:
            auc = peak = fwhm = baseline_roughness = up_area = TTP = gamma_rmse = M = np.nan
        else:
            TimeBetweenVolumes = self.dsc_process.repetition_time
            # TimeBetweenVolumes = 1.56
            # Get peak
            peak = np.max(voxel_signal)

            # Upsample signal:
            original_length = len(voxel_signal)
            upsample_factor = 10
            upsampled_length = original_length * upsample_factor

            time_original = np.arange(original_length) * TimeBetweenVolumes
            time_upsampled = np.linspace(0, time_original[-1], upsampled_length)
            
            interp_func = interp1d(time_original, voxel_signal, kind='linear')
            signal_upsampled = interp_func(time_upsampled)
                
            # Get FWHM
            HM = peak*0.5 # Half maximum
            crossings = np.where(np.diff(np.sign(signal_upsampled-HM)))[0]
            
            # If more than two points crosses the horizontal line of half maximium, the curve should be excluded (by setting fwhm = nan)
            if len(crossings) == 2:
                fwhm = round(time_upsampled[crossings[1]] - time_upsampled[crossings[0]],2)
            else:
                fwhm = np.nan

            if np.isnan(fwhm) or TTP == 0:
                auc = np.nan
                up_area = np.nan
                M = np.nan
                gamma_rmse = np.nan
                baseline_roughness = np.nan
            else:
                # Calc AUC
                auc = simps(voxel_signal, dx=TimeBetweenVolumes) # AUC of bolus passage

                # Get roughness of baseline
                baseline_roughness = simps((np.gradient(np.gradient(voxel_signal[self.dsc_process.baseline_start:self.dsc_process.baseline_end-1], TimeBetweenVolumes)))**2, dx=TimeBetweenVolumes) # Do not include last two baseline points as an AIF might rise early

                # Get up area (Area before GM peak)
                up_area = simps(voxel_signal[self.dsc_process.baseline_start:gm_peak_idx+1], dx=TimeBetweenVolumes)

                gamma_rmse = self._fit_gamma(voxel_signal, sec_pass_gm)
                M = peak/(TTP*fwhm)
    
        return auc, peak, fwhm, baseline_roughness, up_area, TTP, gamma_rmse, M


    def filter_indices(self, df: pd.DataFrame, criteria: list):
        """
        Filter indices based on multiple criteria sequentially.

        Parameters:
        - df: pd.DataFrame, the original DataFrame containing the data.
        - criteria: list of dicts, where each dict contains:
            - 'column': column name to filter on.
            - 'percentile': percentile for filtering.
            - 'operation': operation to use ('>' for greater than, '<' for less than).
        
        Returns:
        - aif_indices: np.ndarray, indices of the DataFrame after applying all filters.
        """
        # Initialize with all indices
        current_indices = np.arange(len(df))
        
        for crit in criteria:
            column = crit['column']
            percentile = crit['percentile']
            operation = crit['operation']
            
            # Select subset of the DataFrame based on current indices
            df_subset = df.iloc[current_indices]
            
            # Calculate the threshold value based on percentile
            if percentile == 'mean':
                threshold = np.mean(df_subset[column])
            else:
                threshold = np.percentile(df_subset[column], percentile)
            
            # Apply the filter operation
            if operation == '>':
                new_indices = current_indices[df_subset[column] > threshold]
            elif operation == '<':
                new_indices = current_indices[df_subset[column] < threshold]
            else:
                raise ValueError("Operation must be '>' or '<'")
            
            # Update current indices
            current_indices = new_indices
        
        return current_indices

    def qc_aif(self, outfile: str = None):
        # Plot new AIF in comparison to the original Prctil AIF
        fig, axs = plt.subplots(4, 1, figsize=(12, 16))
        fig.suptitle(self.dsc_process.sub_id)
        axs[1].set_title('Original Concentration')
        axs[2].set_title('Normalized Concentration')
        axs[3].set_title('Signal')

        # Plot concentration 
        for i in self.final_aif_indices:
            axs[1].plot(self.aifSearchVoxels_conc[i], color='orange', alpha=0.3)
        axs[1].plot(self.final_aif, color='red', linewidth=3, label='Selected AIF')
        axs[1].plot(self.mean_gm_conc, linewidth=2, color='gray', label='Mean GM')
        axs[1].hlines(self.noise_level_conc, -1, len(self.final_aif_signal), linestyle='--', color='red', label='Noise level')
        axs[1].axvline(x=self.dsc_process.baseline_start, linestyle='--', color='blue', alpha=0.5)
        axs[1].axvline(x=self.dsc_process.baseline_end, linestyle='--', color='blue', alpha=0.5)
        axs[1].set_ylabel('Concentration')
        axs[1].set_xlabel('Time (samples)')
        axs[1].set_xlim([-1, len(self.final_aif)])
        axs[1].legend() 

        # Plot normalised concentration 
        for i in self.final_aif_indices:
            axs[2].plot(self.aifSearchVoxels_conc[i]/np.max(self.aifSearchVoxels_conc[i]), color='orange', alpha=0.3)
        axs[2].plot(self.final_aif/np.max(self.final_aif), color='red', linewidth=3, label='Selected AIF')
        axs[2].plot(self.mean_gm_conc/np.max(self.mean_gm_conc), linewidth=2, color='gray', label='Mean GM')
        axs[2].axvline(x=self.dsc_process.baseline_start, linestyle='--', color='blue', alpha=0.5)
        axs[2].axvline(x=self.dsc_process.baseline_end, linestyle='--', color='blue', alpha=0.5)
        axs[2].set_ylabel('Normalized concentration')
        axs[2].set_xlabel('Time (samples)')
        axs[2].set_xlim([-1, len(self.final_aif)])
        axs[2].legend()

        # Plot signal
        for i in self.final_aif_indices:
            axs[3].plot(self.aifSearchVoxels_signal[i], color='orange', alpha=0.3)
        axs[3].plot(self.final_aif_signal, color='red', linewidth=3, label='Selected AIF')
        axs[3].plot(self.mean_gm_signal, linewidth=2, color='gray', label='Mean GM')
        axs[3].hlines(self.noise_level_signal, -1, len(self.final_aif_signal), linestyle='--', color='red', label='Noise level')
        axs[3].axvline(x=self.dsc_process.baseline_start, linestyle='--', color='blue', alpha=0.5)
        axs[3].axvline(x=self.dsc_process.baseline_end, linestyle='--', color='blue', alpha=0.5)
        axs[3].set_ylabel('Signal intensity')
        axs[3].set_xlabel('Time (samples)')
        axs[3].set_xlim([-1, len(self.final_aif_signal)])
        axs[3].set_ylim([0, np.ceil(np.max(self.aifSearchVoxels_signal[self.final_aif_indices]) / 1000) * 1000]) # Max: nearest 1000
        axs[3].legend() 

        plt.tight_layout()

        # Add AIF voxels plot to axs[0]
        x_coords = self.aifSearch_indices[0][self.final_aif_indices]
        y_coords = self.aifSearch_indices[1][self.final_aif_indices]
        z_coords = self.aifSearch_indices[2][self.final_aif_indices]
        self.show_aif_voxs(self.dsc_process.img_data_mean, x_coords, y_coords, z_coords, ax=axs[0], mask=self.aif_search_mask)

        if outfile is not None:
            plt.savefig(outfile)
            plt.close()

    def show_aif_voxs(self, img, x, y, z, ax, mask=None):

        # Group x, y coordinates by their corresponding z values
        z_dict = {}
        for xi, yi, zi in zip(x, y, z):
            if zi not in z_dict:
                z_dict[zi] = []
            z_dict[zi].append((xi, yi))
        
        unique_z = sorted(z_dict.keys())
        n_slices = len(unique_z)

        # Decide margins when plotting differnt number of figure
        # Seems a bit messy, but it works.. :) 
        if n_slices > 5:
            n_rows = 2
            n_cols = int(np.ceil(n_slices / n_rows))
        else:
            n_rows = 1
            n_cols = n_slices

        if n_rows == 2:
            if n_cols > 5: 
                wspace = 0
                hspace = -0.25
                top=0.98
            elif n_cols == 5:
                wspace = -0.40
                hspace = -0.17
                top=0.97
            elif n_cols == 4:
                wspace = -0.75
                hspace = -0.15
                top=0.96
            elif n_cols == 3:
                wspace = -0.79
                hspace = -0.15
                top=0.96
            elif n_cols == 2:
                wspace = -0.82
                hspace = -0.15
                top=0.96
            elif n_cols == 1:
                wspace = 0
                hspace = -0.15
                top=0.96
        else: # n_rows = 1
            wspace = 0
            hspace = 0
            top=0.96


        # Create a GridSpec for the inset Axes within the given ax
        fig = ax.figure
        ax.set_facecolor('black')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set up gridspace
        gs = GridSpec(n_rows, n_cols, figure=fig)
        gs.update(left=ax.get_position().x0, right=ax.get_position().x1,
                top=top, bottom=ax.get_position().y0, wspace=wspace, hspace=hspace)
        
        axes = [fig.add_subplot(gs[r, c]) for r in range(n_rows) for c in range(n_cols)]
        
        for i, zi in enumerate(unique_z):
            if i < len(axes):
                axes[i].imshow(img[:, :, zi], cmap='gray')
                if mask is not None:
                    slice_mask = mask[:, :, zi]
                    slice_mask[np.isnan(img[:, :, zi])] = np.nan
                    axes[i].imshow(slice_mask, cmap='gray', alpha=0.3)
                xs, ys = zip(*z_dict[zi])
                axes[i].plot(ys, xs, '.r')  # Note that y corresponds to column and x to row

                axes[i].axis('off')

        # Hide any unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            axes[j].set_visible(False)

    def _fit_gamma(self, signal, second_pass_gm, plot=False):
  
        s = signal[:second_pass_gm]
        t = np.arange(len(s))
        
        # Define the Gamma Variate Function (Eq. 6 from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6409368/pdf/JBPE-9-69.pdf)
        def gamma_variate(t, t0, tmax, C_max, a):
            # t0: bolus arriaval time
            # tmak: time of peak
            # C_max: max peak of concentration curve
            # a: inflow velocity steepness measure

            # Normalizing time
            t_prime = (t - t0) / (tmax - t0 + 1e-5)
            
            # Replace negative or zero values with a small positive number to avoid invalid computations
            t_prime = np.clip(t_prime, 1e-5, None)

            C_t = C_max * np.exp(a * np.log1p(t_prime - 1) + a * (1 - t_prime))

            # Safeguard against overflow and underflow
            C_t = np.where(np.isfinite(C_t), C_t, 0)

            return C_t
        
        # Define the objective function with weights
        def objective(params, t, s, weights):
            t0, tmax, C_max, a = params
            model = gamma_variate(t, t0, tmax, C_max, a)
            return weights * (model - s)
        
        # Initial guess for the parameters
        initial_guess = (0, np.argmax(s), np.max(s), 10)

        # Define weights, giving more importance to the points near the peak
        weights = np.exp(-0.010 * (t - np.argmax(s))**2)
        bounds = ([0, 0, 0, 0], [1e5, 1e5, 1e5, 1e5])
        
        # Fit the curve using least squares with weights
        result = least_squares(objective, initial_guess, args=(t, s, weights), bounds=bounds)
        fitted_params = result.x

        fitted_t0, fitted_tmax, fitted_C_max, fitted_a = fitted_params

        C_fitted = gamma_variate(t, fitted_t0, fitted_tmax, fitted_C_max, fitted_a)

        residuals = s - C_fitted
        rmse = np.sqrt(np.mean((residuals**2)* weights))

        if plot:
            #Plot the Fitted Curve
            C_fitted = gamma_variate(t, fitted_t0, fitted_tmax, fitted_C_max, fitted_a)
            plt.figure()
            plt.plot(t, s, label='True Curve', color='red')
            plt.plot(t, C_fitted, label='Fitted Curve', color='blue', linestyle='--')
            plt.xlabel('Time (t)')
            plt.ylabel('Concentration (C(t))')
            plt.legend()
            plt.show()

        return rmse
    
    def set_n_aif(self, n_aif):
        self.n_aif = n_aif
