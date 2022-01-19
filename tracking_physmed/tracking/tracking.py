import os, warnings, pickle
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors

from tracking_physmed.utils import (
    get_cmap, get_gaussian_value, get_rectangular_value, get_line_collection,
    plot_color_wheel,
)
from tracking_physmed.gui.get_corners import Corner_Coords

class Tracking(object):

    @property
    def tracking_filepath(self):
        return self._tracking_filepath

    @property
    def tracking_directory(self):
        return self._tracking_directory

    @property
    def video_filepath(self):
        return self._video_filepath

    @property        
    def fps(self):
        """Frames per second from metadata of chosen analysis."""
        return self._fps

    @property
    def pcutout(self):
        return self._pcutout

    @pcutout.setter
    def pcutout(self, value):
        self._pcutout = value

    @property
    def ratio_per_pixel(self):
        return self._ratio_per_pixel

    @property
    def spatial_units(self):
        return self._spatial_units

    @spatial_units.setter
    def spatial_units(self, units):
        
        assert units in ('mm', 'cm', 'm', 'px'), f"Units can only be one of these strings: 'mm', 'cm', 'm' or 'px'. It is '{units}'"
        if units == 'px':
            self._ratio_per_pixel = 1
        else:
            assert self.ratio_cm_per_pixel is not None, 'The cm/px ratio has not yet been set.'
            ratio_factor_dict = {'mm':10, 'cm': 1, 'm': 0.1}
            self._ratio_per_pixel = self.ratio_cm_per_pixel * ratio_factor_dict[units]

        self._spatial_units = units

    def __init__(self, filename=None, video_filename=None) -> None:

        if not os.path.isfile(filename):
            raise FileNotFoundError('Check filename and make sure the file exists.')

        filename = Path(filename)
        self._tracking_filepath = filename
        self._tracking_directory = self._tracking_filepath.parent

        self._video_filepath = video_filename
        if video_filename is None:
            self._video_filepath = self.tracking_directory.joinpath(self.tracking_filepath.stem + '_labeled.mp4')

        if not os.path.isfile(self._video_filepath):
            warnings.warn(f'Tried to guess video filepath as {self._video_filepath}, but file does not exist.\n'+ \
                'Use self.set_video_filepath(filename) for the animations to work with the right video.', category=UserWarning)
            self._video_filepath = None
        else:
            # There is video filepath
            self._video_filepath = Path(self._video_filepath)

        self.colormap = 'plasma'
        self._pcutout = 0.8
        self._load_Dataframe()

    def _load_Dataframe(self):

        self.Dataframe = pd.read_hdf(self.tracking_filepath)
        if 'filtered' in str(self.tracking_filepath):
            self.metadata_filepath = str(self.tracking_filepath).split('_filtered')[0]+'_meta.pickle'
        else:
            self.metadata_filepath = str(self.tracking_filepath).split('.')[0]+'_meta.pickle'

        self.metadata = pd.read_pickle(self.metadata_filepath)
        self.scorer = self.metadata['data']['Scorer']

        self.nframes = self.Dataframe.shape[0]
        self.bodyparts = self.metadata['data']['DLC-model-config file']['all_joints_names']
        self._fps = self.metadata['data']['fps']
        colors =   get_cmap(len(self.bodyparts), name=self.colormap)
        self.colors = {self.bodyparts[i]: colors(i) for i in range(len(self.bodyparts))}

        try:
            # checking if coordinates for corners have already been set
            # if not, calls function to do it
            self.metadata['data']['corner_coords']
        except KeyError:
            self.set_corner_coords()

        self._get_cm2px_ratio(w=100, h=100)
        self._spatial_units = 'cm'
        self._ratio_per_pixel = self.ratio_cm_per_pixel

    def set_corner_coords(self, coord_list = []):
        """If coord_list is not given, it calls Corner_Coords class GUI so the user can label the four corners and the Tracking class is able to calculate the ratio px/cm. It rights the corner coordinates in the metadata pickle file.

        Parameters
        ----------
        coord_list : list, optional
            Should be a list of [x, y] coordinates for the top left, top right, bottom left and bottom right corners such that coord_list = [[tl_x, tl_y], [tr_x, tr_y], ...], by default []
        """
        if coord_list:
            self._write_corner_coords(coord_list)
        else:
            x_crop = self.metadata['data']['cropping_parameters'][:2]
            y_crop = self.metadata['data']['cropping_parameters'][2:]
            self.corner_coords = Corner_Coords(self.video_filepath,
                                               function_after_done=self._write_corner_coords,
                                               x_crop=x_crop,
                                               y_crop=y_crop)

    def _write_corner_coords(self, coords_list):
        """Writes corner coordinates in the metadata of the analysis so it is there for the next time
        and set_corner_coords does not need to be called again.
        """
        with open(self.metadata_filepath, 'wb') as f:
            try:
                self.metadata['data']['corner_coords'] = {}
                self.metadata['data']['corner_coords']['top_left'] = np.array(coords_list[0])
                self.metadata['data']['corner_coords']['top_right'] = np.array(coords_list[1])
                self.metadata['data']['corner_coords']['bottom_left'] = np.array(coords_list[2])
                self.metadata['data']['corner_coords']['bottom_right'] = np.array(coords_list[3])
                print("Corner coordinates saved correctly!")
            except AttributeError:
                print("Corner coordinates were not saved!")
                pass
            pickle.dump(self.metadata, f)

    def _get_cm2px_ratio(self, w, h):
        """Helper function that calculates the cm/px ratio from the user inputs of the corners of the arena.

        Parameters
        ----------
        w : width in cm
            Distance between right and left corners. The default is 100.
        h : height in cm
            Distance between top and bottom corners. The default is 100.

        Returns
        -------
        float
            Returns the ratio of cm/px of the images being analysed.
        """
        ## TO SEE IF W AND H ARE NOT THE SAME!!
        
        try:
            estimates = np.empty(4)
            estimates[0] = np.sqrt(np.sum((self.metadata['data']['corner_coords']['top_right'] \
                 - self.metadata['data']['corner_coords']['top_left'])**2))
            estimates[1] = np.sqrt(np.sum((self.metadata['data']['corner_coords']['bottom_right'] \
                 - self.metadata['data']['corner_coords']['bottom_left'])**2))
            
            estimates[2] = np.sqrt(np.sum((self.metadata['data']['corner_coords']['top_left'] \
                 - self.metadata['data']['corner_coords']['bottom_left'])**2))
            estimates[3] = np.sqrt(np.sum((self.metadata['data']['corner_coords']['top_right'] \
                 - self.metadata['data']['corner_coords']['bottom_right'])**2))
        
            w_estimate = w / estimates[:2].mean()
            h_estimate = h / estimates[2:].mean()
            self.ratio_cm_per_pixel = (w_estimate + h_estimate)/2
            return self.ratio_cm_per_pixel
        
        except KeyError:
            print('Ratio cm/px not yet calculated. See function self.set_corner_coords.')

    def get_vector_from_two_labels(self, label0, label1):
        """Gets the vector 'label0'->'label1' by simple subtraction label1 - label0.  

        Parameters
        ----------
        label0 : str
            Label where the vector will start.
        label1 : str
            Label where the vector will finish.

        Returns
        -------
        vec_x : numpy.ndarray
            Vector distance in the x coordinate. label1_x - label0_x
        vec_y : numpy.ndarray
            Vector distance in the y coordinate. label1_y - label0_y
        """

        vec_x = self.Dataframe[self.scorer][label1]['x'] - self.Dataframe[self.scorer][label0]['x']
        vec_y = self.Dataframe[self.scorer][label1]['y'] - self.Dataframe[self.scorer][label0]['y']
        return vec_x.to_numpy(), vec_y.to_numpy()
    
    def get_direction_array(self, label0='neck', label1='probe', mode='deg'):
        """Gets the direction vector 'label0'->'label1' by simple subtraction label1 - label0.
        Default vector is 'neck'->'probe'. This can be used to get the head direction of the animal, for example.

        Parameters
        ----------
        label0 : str, optional
            Label where the vector will start. The default is 'neck'.
        label1 : str, optional
            Label where the vector will finish. The default is 'probe'.

        Returns
        -------
        direction array : numpy.ndarray
            In degrees, if mode asks for it, otherwise in radians.

        """
        
        hd_x, hd_y = self.get_vector_from_two_labels(label0=label0, label1=label1)

        resp_in_rad = np.arctan2(hd_y*(-1), hd_x) # multiplication by -1 needed because of video x and y directions
        resp_in_rad[resp_in_rad < 0] += 2*np.pi
        if mode in ('deg', 'degree'):
            return np.degrees(resp_in_rad)
        return resp_in_rad

    def get_degree_interval_hd(self, deg, only_running_bouts = False):
        """Gets an array where the direction array (head direction here) is modulated by a guassian function centered in `deg`.

        Parameters
        ----------
        deg : int or float
            Head direction in degrees, between 0 and 360.
        only_running_bouts : bool, optional
            [description], by default False

        Returns
        -------
        [type]
            [description]
        """
        hd_deg = self.get_direction_array(label0='neck', label1='probe', mode='deg')
        
        time_array = np.array(self.Dataframe.index) / self.fps
        index = self.Dataframe[self.scorer]['probe']['likelihood'].values >= self.pcutout
        
        sigma = 20
        
        if deg < 60:
            hd_deg = np.where(hd_deg > 300, 360 - hd_deg, hd_deg)
        
        if deg > 300:
            hd_deg = np.where(hd_deg < 60, hd_deg + 360, hd_deg)
            
        tmp = hd_deg - deg
        
        hd_array = get_gaussian_value(tmp,sigma)
        
        if only_running_bouts == True:
            
            self.get_running_bouts()
            hd_bouts = [x for x in np.split(np.where(self.running_bouts, hd_array, 0),self.final_change_idx+1) if x[0]!=0]
            time_bouts = [x for x in np.split(np.where(self.running_bouts, time_array, 0),self.final_change_idx+1) if x[1]!=0]
            index_bouts = [x for x in np.split(np.where(self.running_bouts, index, 0),self.final_change_idx+1) if x[0]!=0]
            
            return hd_bouts, time_bouts, index_bouts
        
        return hd_array, time_array, index

    def get_position_x(self,bodypart,pcutout=.8):
        """
        Simple function to get x values for bodypart.

        Parameters
        ----------
        bodypart : str
        pcutout : float, optional
            Between 0 and 1. The default is .8.

        Returns
        -------
        x_bp : numpy.ndarray
            Pixel values in x for bodypart.
        # index : numpy.ndarray
            Index where p-value > pcutout is True, index is False otherwise.

        """
        index = self.Dataframe[self.scorer][bodypart]['likelihood'].values > pcutout
        x_bp = self.Dataframe[self.scorer][bodypart]['x'].values - self.metadata['data']['corner_coords']['top_left'][0]
        time_array = np.array(self.Dataframe.index) / self.fps
        
        return x_bp, time_array, index

    def get_position_y(self,bodypart,pcutout=0.8):
        """
        Simple function to get y values for bodypart.

        Parameters
        ----------
        bodypart : str
        pcutout : float, optional
            Between 0 and 1. The default is .8.

        Returns
        -------
        y_bp : numpy.ndarray
            Pixel values in y for bodypart.
        index : numpy.ndarray
            Index where p-value > pcutout is True, index is False otherwise.

        """
        
        index = self.Dataframe[self.scorer][bodypart]['likelihood'].values > pcutout
        y_bp = self.Dataframe[self.scorer][bodypart]['y'].values - self.metadata['data']['corner_coords']['top_left'][1]
        time_array = np.array(self.Dataframe.index) / self.fps
        
        return y_bp, time_array, index

    def get_distance_between_frames(self,
                                    bodypart='body',
                                    backup_bps=['probe']
                                    ):        
        """
        Get distance from one frame to another for the specific bodypart along the whole analysis.

        Parameters
        ----------
        bodypart : str, optional
            The default is 'body'.

        Returns
        -------
        numpy.ndarray, first return
            Array of size index[True]. This means only the index which likelihood is above the given threshold.
            First values is set to 0 so that the returned array has the same size of self.nframes.
            
        numpy.ndarray, second return
            Array of size index[True].
            Distance between (x,y) coordinates from one frame to another.
            First values is set to 0 so that the returned array has the same size of index[True].

        """
        x_pts = self.get_position_x(bodypart=bodypart)[0]
        y_pts = self.get_position_y(bodypart=bodypart)[0]
    
        dist_in_px = np.sqrt(np.diff(x_pts)**2 + np.diff(y_pts)**2)

        return np.insert(dist_in_px, 0, 0)

    def get_speed(self,
                  bodypart='body',
                  smooth=True,
                  speed_cutout=0,
                  only_running_bouts=False,
                  normalized=False,
                  **kwargs):
        
        dist_in_px = self.get_distance_between_frames(bodypart=bodypart)
        speed_in_px_per_second = dist_in_px * self.fps
        
        index = self.Dataframe[self.scorer][bodypart]['likelihood'].values >= self.pcutout

        time_array = np.array(self.Dataframe.index) / self.fps
        
        speed_units = self.spatial_units + '/s'
        speed_array = speed_in_px_per_second * self.ratio_per_pixel 
            
        if smooth:
            window = signal.gaussian(M=101,std=6)
            window /= sum(window)
            speed_array[~index] = np.nan
            speed_array = np.convolve(window, speed_array, 'same')
            speed_array[speed_array < speed_cutout] = 0
            
        if only_running_bouts == True:
            self.get_running_bouts(speed_array=speed_array, time_array=time_array)
            speed_bouts = [x for x in np.split(np.where(self.running_bouts, speed_array, 0),self.final_change_idx+1) if x[0]!=0]
            time_bouts = [x for x in np.split(np.where(self.running_bouts, time_array, 0),self.final_change_idx+1) if x[1]!=0]
            index_bouts = [x for x in np.split(np.where(self.running_bouts, index, 0),self.final_change_idx+1) if x[0]!=0]

            return speed_bouts, time_bouts, index_bouts, speed_units
        else:    
            return speed_array, time_array, index, speed_units

    def get_binned_position(self):
        pass    

    def get_infos(self):
        
        info_dict = {}
        info_dict['total_time'] = self.nframes / self.fps
        
        speed, _, _, _ = self.get_speed(bodypart='body', smooth=True)
        speed_bouts, time_bouts, _, speed_units = self.get_speed(only_running_bouts=True)
        bout_lenghts = [t_bout[-1] - t_bout[0] for t_bout in time_bouts]
        info_dict['total_running_time'] = sum(bout_lenghts)
        info_dict['timefraction_of_running'] = info_dict['total_running_time'] / info_dict['total_time']
        info_dict['mean_running_speed'] = np.concatenate(speed_bouts).mean()
        info_dict['mean_speed'] = speed.mean()
        print("--------------------------------------------------------------\n"+
              f"Total tracking time: {info_dict['total_time']} s\n"+
              f"Total running time: {info_dict['total_running_time']:.2f}\n"+
              f"Exploration ratio (total time / running time): {info_dict['timefraction_of_running']:.3f}\n"+
              f"Mean running speed (only running periods): {info_dict['mean_running_speed']:.2f} {speed_units}\n"+
              f"Mean running speed: {info_dict['mean_speed']:.2f} {speed_units}\n"+
              "--------------------------------------------------------------")
        return info_dict

