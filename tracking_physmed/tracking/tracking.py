import os, warnings
from pathlib import Path
import pandas as pd

from tracking_physmed.utils.utils import get_cmap

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
        """Get frames per second from metadata of chosen analysis."""
        return self._fps

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

        # try:
        #     # checking if coordinates for corners have already been set
        #     # if not, calls function to do it
        #     self.metadata['data']['corner_coords']
        # except KeyError:
        #     self.set_corner_coords()

        # self._get_cm2px_ratio(w=100, h=100)
        # self._spatial_units = 'cm'
        # self._ratio_per_pixel = self.ratio_cm_per_pixel
