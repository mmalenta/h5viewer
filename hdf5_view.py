import argparse as ap
import cmd
import glob
import h5py as h5
import io
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image

# TODO: add mask display

class Interpreter(cmd.Cmd):

    def __init__(self, h5viewer):
        self.intro = "MeerTRAP HDF5 visualisation tool"
        self.prompt = ">> "
        super(Interpreter, self).__init__()
        self.__h5viewer = h5viewer

    def do_clean(self, arg):

        self.__h5viewer.Reset()

    def do_dedisp(self, arg):
        dm = float(arg)
        print("Dedispersing the data to DM of %f..." % (dm))
        self.__h5viewer.Dedisperse(dm)

    def do_list(self, arg):
        """List the contents of the HDF5 archive"""
        self.__h5viewer.List()

    def do_load(self, arg):

        self.__h5viewer.Load(arg)

    def do_print(self, arg):
        """Print requested information. Relevant options:
        * fil - prints filterbank header information
        * cand - prints candidate information
        * fetch - prints fetch classification
        """

        if arg not in ["fil", "cand", "fetch"]:
            print("Did not recognise option '%s'" % (arg))
        else:
            self.__h5viewer.PrintInfo(arg)

    def do_plot(self, arg):
        """Plot the data. Relevant options:
        * jpg - plots the candidate analysis jpg
        * fil - plots the original filterbank file
        * mask - plots the applied mask
        * fetch dm/freq - plots dm-time or frequency time FETCH output (if no dm/freq provided, it will plot a combined plot)

        Additional options (available for certain types of plots only, as indicated):

        * masked (fil) - plots the mask file
        * normalised (fil) - normalise the data before plotting (subtract the mean and divide by standard deviations on channel-by-channel basis)
        * all (jpg, fil, fetch) - plot all files in the current directory
        * save (jpg, fil, fetch) - save the plot(s) to disk (default for the 'all' option)
        """

        split_arg = arg.split()
        
        # Parse the plot type
        allowed_types = ['fil', 'jpg', 'fetch', 'mask']
        allowed_passed = [a in split_arg for a in allowed_types]

        if allowed_passed.count(True) != 1:
            print("Need to pass a single plot type from the following list: [jpg, fil, fetch]")
            return None

        plot_type = allowed_types[allowed_passed.index(True)]

        # Parse additional options
        mask_data = False
        plot_all = False
        save_plots = False
        normalise = False

        def check_option(option: str) -> bool:

            if option in split_arg:
                return True
            else:
                return False

        mask_data = check_option('masked')
        normalise = check_option('normalised')
        plot_all = check_option('all')
        save_plots = check_option('save')
        # NOTE: We do not want to potentially display hundreds of candidate plots - just save then to disk
        if plot_all == True:
            save_plots = True

        if plot_type == "jpg":
        
            self.__h5viewer.PlotJPG()
        
        elif plot_type == "fil":
        
            self.__h5viewer.PlotFil(mask_data=mask_data, normalise=normalise, plot_all=plot_all, save_plots=save_plots)
        
        elif plot_type == "fetch":
            
            allowed_types = ['dm', 'freq']
            allowed_passed = [a in split_arg for a in allowed_types]

            if allowed_passed.count(True) > 1:
                print("Please pass one or less types from the following list: [dm, freq]\nIf no types are passed, a combined plot will be generated")
                return None

            if allowed_passed.count(True) == 0:
                axis = "combined"
            else:
                axis = allowed_types[allowed_passed.index(True)]

            self.__h5viewer.PlotFetch(axis, plot_all=plot_all, save_plots=save_plots)

        elif plot_type == "mask":
            self.__h5viewer.PlotMask()
        else:
            print("I did not recognise the option %s" % (arg))

    def do_quit(self, arg):
        print("Quitting...")
        return True

    def do_set(self, arg):

        split_arg = arg.split()

        if len(split_arg) == 2:
            self.__h5viewer.Set(*split_arg)

    def do_summary(self, arg):

        split_arg = arg.split()

        if len(split_arg) == 3 and (split_arg[1] in ["aa", "fetch", "combined", "all"]) and (split_arg[2] in ["list", "plot"]):
            self.__h5viewer.Summary(*split_arg)
        else:
            print("Did not recognise option '%s'" % arg)

    #def precmd(self, arg):
    #   print('\n')
        #return arg.lower()

class H5Viewer:

    def __init__(self, config):

        self._verbose = config['verbose']
        self._file_name = config['file']

        self._base_dir = './'
        self._file = None

        if self._file_name != None:
            try:
                self._file = h5.File(self._file_name, 'r')
            except OSError:
                print("File '%s' not found " % (self._file_name))
                quit()

        self._fil_array = np.empty(0)
        self._fil_header = {}
        self._cand_info = {}
        self._fetch_info = {}

    def CheckKeys(self, top_key):

        if type(self._file[top_key]) != h5._hl.dataset.Dataset:
            for key in self._file[top_key].keys():
                cur_key = top_key + key

                if type(self._file[cur_key]) == h5._hl.dataset.Dataset:
                    print('d\'', cur_key)
                    print('\tShape: ', self._file[cur_key].shape)
                    print('\tType: ', self._file[cur_key].dtype)
                    print('\tCompression:', self._file[cur_key].compression)
                else:
                    print('g\'', top_key + key)
                
                if len(self._file[cur_key].attrs) != 0:
                    print("\tAttributes: ")
                    for attr, value in self._file[cur_key].attrs.items():
                        print('\t' + attr + ': ' + str(value))

                self.CheckKeys(top_key + key + '/') 


    def List(self):

        print("Contents of the HDF5 file:")
        self.CheckKeys('/')

    def Load(self, file_name, join_base=True):

        if join_base == True:
            self._file_name = os.path.join(self._base_dir, file_name)
        else:
            self._file_name = file_name
        
        if self._file_name != None:
            try:
                self._file = h5.File(self._file_name, 'r')
            except OSError:
                print("File '%s' not found " % (self._file_name))

    def ReadFil(self):

        # Make sure we only do this once
        if self._fil_array.size == 0:
            print("Reading the filterbank file...")
            # Automatically decompressed if needed
            fil_dataset = self._file['/cand/search/filterbank/data']
            nchans = self._file['/cand/search/filterbank'].attrs['nchans']
            header_len = self._file['/cand/search/filterbank'].attrs['head_len']
            fil_array = np.reshape(np.array(fil_dataset[:], dtype='B')[header_len:], (-1, nchans)).T
            self._fil_array = np.copy(fil_array)
            mean = np.mean(self._fil_array, axis=1)
            std = np.std(self._fil_array, axis=1)

            self._fil_array = self._fil_array - mean[:, np.newaxis]
            std[std == 0] = 1
            self._fil_array = np.divide(self._fil_array, std[:, np.newaxis])

        return self._fil_array

    def ReadInfo(self, type):

        if len(self._fil_header) == 0:
            self._fil_header['tstart'] = self._file['/cand/search/filterbank'].attrs['tstart']
            self._fil_header['tsamp'] = self._file['/cand/search/filterbank'].attrs['tsamp']
            self._fil_header['nbits'] = self._file['/cand/search/filterbank'].attrs['nbits']
            self._fil_header['nchans'] = self._file['/cand/search/filterbank'].attrs['nchans']
            self._fil_header['fch1'] = self._file['/cand/search/filterbank'].attrs['fch1']
            self._fil_header['foff'] = self._file['/cand/search/filterbank'].attrs['foff']
            self._fil_header['src_raj'] = self._file['/cand/search/filterbank'].attrs['src_raj']
            self._fil_header['src_dej'] = self._file['/cand/search/filterbank'].attrs['src_dej']

        if len(self._cand_info) == 0:
            self._cand_info['mjd'] = self._file['/cand/search'].attrs['mjd']
            self._cand_info['dm'] = self._file['/cand/search'].attrs['dm']
            self._cand_info['snr'] = self._file['/cand/search'].attrs['snr']
            self._cand_info['width'] = self._file['/cand/search'].attrs['width']
            self._cand_info['beam'] = self._file['/cand/search'].attrs['beam']
            self._cand_info['beam_type'] = self._file['/cand/search'].attrs['beam_type']
            self._cand_info['beam_ra'] = self._file['/cand/search'].attrs['beam_ra']
            self._cand_info['beam_dec'] = self._file['/cand/search'].attrs['beam_dec']

        if len(self._fetch_info) == 0:
            self._fetch_info['label'] = self._file['/cand/fetch'].attrs['label']
            self._fetch_info['probability'] = self._file['/cand/fetch'].attrs['probability']

        if type == "fil":
            return self._fil_header
        elif type == "cand":
            return self._cand_info
        elif type == "fetch":
            return self._fetch_info

    def Reset(self):

        if type(self._file) == h5._hl.files.File:
            print("Closing the H5 file...")
            self._file.close()
        self._file = None
        self._fil_array = np.empty(0)
        self._fil_header = {}
        self._cand_info = {}
        self._fetch_info = {}

    def Dedisperse(self, dm=-1.0):

        if dm == -1.0:
            dm = self._file['/cand/search/'].attrs['dm']
            print("Using the detection DM of %.2f" % (dm))
        else:
            print("Dedispersing to the DM of %.2f" % (dm))

        fil_array = self.ReadFil()
        nchans = self._file['/cand/search/filterbank'].attrs['nchans']

    def PlotMask(self):

        mask_dataset = self._file['/cand/search/filterbank/mask']
        fig = plt.figure(figsize=(10,4))
        ax = fig.gca()
        ax.plot(mask_dataset, linewidth=0.5)
        ax.set_yticks([0, 1])
        ax.set_xlabel('Channel number (0 - highest frequency)')
        plt.show(block=False)

    def PlotJPG(self):

        plot_dataset = self._file['/cand/search/plot/jpg']
        im = Image.open(io.BytesIO(plot_dataset[:]))
        im.show()

    def PlotFil(self, mask_data=False, normalise=False, plot_all=False, save_plots=False):

        cand_files = [self._file_name]

        if plot_all:
            cand_files = sorted(glob.glob(os.path.join(self._base_dir, '5*.hdf5')))

        for cand_file in cand_files:
            print(cand_file)
            self.Load(cand_file, join_base=False)
            fil_array = self.ReadFil()

            fmt = lambda x: "{:.2f}".format(x)

            freq_pos = np.linspace(0, self._file['/cand/search/filterbank'].attrs['nchans'], num=5)
            freq_label = [fmt(label) for label in (self._file['/cand/search/filterbank'].attrs['fch1'] + freq_pos * self._file['/cand/search/filterbank'].attrs['foff']) ]

            time_pos = np.linspace(0, fil_array.shape[1], num=5)
            time_label = [fmt(label) for label in (time_pos * self._file['/cand/search/filterbank'].attrs['tsamp'])]

            if mask_data:
                mask_dataset = np.asarray(self._file['/cand/search/filterbank/mask'])
                fil_array = fil_array * mask_dataset[:, np.newaxis]

            if normalise:
                band_mean = np.mean(fil_array, axis=1)
                band_std = np.std(fil_array, axis=1)
                band_std[band_std == 0] = 1
                fil_array = np.nan_to_num((fil_array - band_mean[:, np.newaxis]) / band_std[:, np.newaxis])

            fig = plt.figure(figsize=(9,6))
            ax = fig.gca()
            ax.imshow(fil_array, aspect='auto', cmap='binary', interpolation='none')
            ax.set_xticks(time_pos)
            ax.set_xticklabels(time_label, fontsize=8)
            ax.set_xlabel('Time [s]')

            ax.set_yticks(freq_pos)
            ax.set_yticklabels(freq_label, fontsize=8)
            ax.set_ylabel('Frequency [MHz')

            if save_plots:
                plot_name = os.path.join(self._base_dir, 'mjd_' + str(self._file['/cand/search'].attrs['mjd']) + '_dm_' + str(self._file['/cand/search'].attrs['dm']) + '_beam_' + str(self._file['/cand/search'].attrs['beam']) + str(self._file['/cand/search'].attrs['beam_type']) + '_fil.png')
                plt.savefig(plot_name)
                fig.clear()
                plt.close(fig)
            else:
                plt.show(block=False)

            if plot_all:
                self.Reset()

    def PlotFetch(self, axis, plot_all=False, save_plots=False):


        cand_files = [self._file_name]

        if plot_all:
            cand_files = sorted(glob.glob(os.path.join(self._base_dir, '5*.hdf5')))

        for cand_file in cand_files:
            print(cand_file)
            self.Load(cand_file, join_base=False)

            fig = None
            ax = None

            if axis == 'combined':
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))

                ax[0].imshow(np.array(self._file['/cand/fetch/dm_time']), aspect='auto', cmap='binary', interpolation='none')
                ax[1].imshow(np.array(self._file['/cand/fetch/freq_time']).T, aspect='auto', cmap='binary', interpolation='none')
            else:
                fig = plt.figure(figsize=(5,5))
                ax = fig.gca()

                plot_dataset = np.array(self._file['/cand/fetch/' + axis + '_time'])
                plot_dataset = plot_dataset.T
                ax.imshow(plot_dataset, aspect='auto', cmap='binary', interpolation='none')
                
            if axis != 'combined':
                cand_label = self._file['/cand/fetch'].attrs['label']
                cand_prob = self._file['/cand/fetch'].attrs['probability']
                ax.text(0.1, 0.95, 'Label: ' + str(cand_label), color='firebrick', fontweight='bold',  transform=ax.transAxes)
                ax.text(0.1, 0.9, 'Probability: ' + "{:.4f}".format(cand_prob * 100.0) + "%", color='firebrick', fontweight='bold', transform=ax.transAxes)
            else:
                cand_dm = self._file['/cand/search'].attrs['dm']
                ax[0].text(0.0, 1.05, 'DM: ' + "{:.2f}".format(cand_dm), color='black', fontweight='bold', transform=ax[0].transAxes)
                ticks_labels = ["{:.2f}".format(dm) for dm in np.linspace(0, 2 * cand_dm, 6, dtype=np.float32)]
                ax[0].set_xlabel('Time sample')
                ax[0].set_yticks(np.linspace(0, 256, 6))
                ax[0].set_yticklabels(ticks_labels)
                ax[0].set_ylabel(r'Trial DM [pc $cm^{-3}$]')
                ax[1].set_xlabel('Time sample')
                ax[1].set_ylabel('Frequency channel')
            if save_plots:
                plot_name = os.path.join(self._base_dir, 'mjd_' + str(self._file['/cand/search'].attrs['mjd']) + '_dm_' + str(self._file['/cand/search'].attrs['dm']) + '_beam_' + str(self._file['/cand/search'].attrs['beam']) + str(self._file['/cand/search'].attrs['beam_type']) + '_fetch_' + axis + '.png')
                plt.savefig(plot_name)
                fig.clear()
                plt.close(fig)
            else:
                plt.show(block=False)

            if plot_all:
                self.Reset()

    def PrintInfo(self, type):

        info = self.ReadInfo(type)

        print("%s information:" % (type.capitalize()))

        for key in info:
            print("%s: %s" % (key, str(info[key]) )) 

    def Set(self, key, value):

        if key in ["directory", "dir"]:
            self._base_dir = value
        else:
            print("Unrecognised key %s" % (key))

    def Summary(self, directory, source, disp):

        directory = os.path.join(self._base_dir, directory)

        if (os.path.isdir(directory) != True):
            print("Directory %s does not exist!" % (directory))
            return False

        cand_files = sorted(glob.glob(os.path.join(directory, 'beam*/5*.hdf5')))
        
        if disp == "list":
            negative_labels = 0
            positive_labels = 0
            for cand_file in cand_files:
                print(cand_file[cand_file.rfind('/') + 1:])
                with h5.File(cand_file, 'r') as h5_file:
                    print("\tFilterbank file: %s" % (h5_file['/cand/search/filterbank'].attrs['filterbank']))
                    if source == "aa" or source == "all":
                        print("\tMJD: %.6f" % (h5_file['/cand/search'].attrs['mjd']))
                        print("\tDM: %.2f" % (h5_file['/cand/search'].attrs['dm']))
                        print("\tWidth: %.4f" % (h5_file['/cand/search'].attrs['width']))
                        print("\tSNR: %.4f" % (h5_file['/cand/search'].attrs['snr']))

                    if source == "fetch" or source == "all":
                        print("\tLabel: %d" % (h5_file['/cand/fetch'].attrs['label']))
                        if h5_file['/cand/fetch'].attrs['label'] == 1:
                            positive_labels = positive_labels + 1
                        else: 
                            negative_labels = negative_labels + 1
                        print("\tProbability: %.4f" % (h5_file['/cand/fetch'].attrs['probability']))

            print("Labels summary:")
            print("Label 1: %d candidates" % (positive_labels))
            print("Label 0: %d candidates" % (negative_labels))

        elif disp == "plot":

            if source == "aa" or source == "all":

                mjds = []
                dms = []
                widths = []
                snrs = []

                for cand_file in cand_files:
                    with h5.File(cand_file, 'r') as h5_file:
                        mjds.append(h5_file['/cand/search'].attrs['mjd'])
                        dms.append(h5_file['/cand/search'].attrs['dm'] + 1)
                        widths.append(h5_file['/cand/search'].attrs['width'])
                        snrs.append(h5_file['/cand/search'].attrs['snr'])

                plot_pad = 10.0 / 86400 

                fig = plt.figure(figsize=(9, 6))
                ax = fig.gca()
                scatter = ax.scatter(x = mjds, y = dms, s = widths * 10, c = snrs)
                ax.ticklabel_format(useOffset=False)
                ax.set_yscale("log")
                ax.set_xlim([min(mjds) - plot_pad, max(mjds) + plot_pad])

                mjd_values = np.linspace(min(mjds), max(mjds), 5)
                fmt = lambda x: "{:.6f}".format(x)
                mdj_strings = [ fmt(label) for label in mjd_values ]

                ax.set_xticks(mjd_values)
                ax.set_xticklabels(mdj_strings, fontsize=8)
                ax.set_xlabel('MJD', fontsize=10)
                ax.set_ylabel('DM + 1', fontsize=10)
                cbar = fig.colorbar(scatter)
                cbar.set_label('SNR')
                plt.show(block=False)

            if source == "fetch" or source == "all":
                
                probs = []

                for cand_file in cand_files:
                    with h5.File(cand_file, 'r') as h5_file:
                        probs.append(h5_file['/cand/fetch'].attrs['probability'])

                fig = plt.figure(figsize=(9,6))
                ax = fig.gca()                
                ax.hist(probs, bins=100)
                ax.set_xlabel('Probability', fontsize=10)
                ax.set_ylabel('# counts', fontsize=10)
                ax.set_xlim([-0.1, 1.1])
                plt.show(block=False)

            if source == "combined" or source == "all":

                mjds = []
                dms = []
                widths = []
                snrs = []
                labels = []

                cmap = matplotlib.colors.ListedColormap(['firebrick', 'darkgreen'])
                boundaries = [0, 1]
                norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

                for cand_file in cand_files:
                    with h5.File(cand_file, 'r') as h5_file:
                        mjds.append(h5_file['/cand/search'].attrs['mjd'])
                        dms.append(h5_file['/cand/search'].attrs['dm'] + 1)
                        widths.append(h5_file['/cand/search'].attrs['width'])
                        snrs.append(h5_file['/cand/search'].attrs['snr'])
                        labels.append(h5_file['/cand/fetch'].attrs['label'])

                plot_pad = 10.0 / 86400 

                fig = plt.figure(figsize=(9, 6))
                ax = fig.gca()
                scatter = ax.scatter(x = mjds, y = dms, s = snrs * 10, c = labels, cmap=cmap, alpha=0.5)
                ax.ticklabel_format(useOffset=False)
                ax.set_yscale("log")
                ax.set_xlim([min(mjds) - plot_pad, max(mjds) + plot_pad])
                
                mjd_values = np.linspace(min(mjds), max(mjds), 5)
                fmt = lambda x: "{:.6f}".format(x)
                mdj_strings = [ fmt(label) for label in mjd_values ]

                ax.set_xticks(mjd_values)
                ax.set_xticklabels(mdj_strings, fontsize=8)
                ax.set_xlabel('MJD', fontsize=10)
                ax.set_ylabel('DM + 1', fontsize=10)
                ax.text(0.1, 0.95, 'Size -> SNR', transform=ax.transAxes)
                cbar = fig.colorbar(scatter)
                cbar.set_ticks([0.25, 0.75])
                cbar.set_ticklabels(['Label 0', 'Label 1'])
                plt.show(block=False)
                
    def __GetHeaderValue(self, file, key, type):
        to_read = len(key)
        step_back = -1 * (to_read - 1)

        while(True):
            read_key = str(file.read(to_read).decode('iso-8859-1'))
            if (read_key == key):

                if (type == "int"):
                    value, = struct.unpack('i', file.read(4))

                if (type == "double"):
                    value, = struct.unpack('d', file.read(8))

                if (type == "str"):
                    to_read, = struct.unpack('i', file.read(4))
                    value = str(file.read(to_read).decode('iso-8859-1'))

                file.seek(0)
                break
            file.seek(step_back, 1)

        return value

def main():
    parser = ap.ArgumentParser(description="View contents of MeerTRAP HDF5 files",
                                usage="%(prog)s <options>",
                                epilog="For any bugs, please start an issue at https://gitlab.com/MeerTRAP/frb_plots")

    parser.add_argument("-v", "--verbose", help="Enable verbose mode", action="store_true", required=False)
    parser.add_argument("-f", "--file", help="Input HDF5 file", required=False, type=str)
    parser.add_argument("-l", "--list", help="List the contents of the HDF5 file", action="store_true", required=False)
    parser.add_argument("-i", "--interactive", help="Enable interactive mode", action="store_true", required=False)

    arguments = parser.parse_args()

    configuration = {
        'verbose': arguments.verbose,
        'file': arguments.file
    }

    h5viewer = H5Viewer(configuration)

    interpreter = Interpreter(h5viewer)
    interpreter.cmdloop()

if __name__ == "__main__":
    main()