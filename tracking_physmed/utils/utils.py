import matplotlib as plt
import numpy as np
from matplotlib import colors

def get_cmap(n, name='hsv'):
    """Get a colormap instance with n numbers of entries.

    Parameters
    ----------
    n : int
        Number of entries desired in the lookup table.
    name : str, optional
        Must be a standard matplotlib colormap name. The default is 'hsv'.

    Returns
    -------
    matplotlib.colors.Colormap instance

    """
    return plt.cm.get_cmap(name, n)

def get_line_collection(x_array, y_array, index):
    """Gets collection of arrays for each segmente of x_array and y_array.
    Returns this collection where index == True.

    Parameters
    ----------
    x_array : array
    y_array : array
    index : list or array of bool

    Returns
    -------
    array

    """
    if type(x_array) is not list:
        x_array = [x_array]
        y_array = [y_array]
        index = [index]
        
    segments = []
    segment_index = []
    for x, y, idx in zip(x_array, y_array, index):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments.append(np.concatenate([points[:-1], points[1:]], axis=1))
        segment_index.append(np.logical_and(idx[1:],idx[:-1]))

    segments = np.concatenate(segments, axis=0)
    segment_index = np.concatenate(segment_index, axis=0)
    return segments[segment_index]
    
def get_gaussian_value(n, sigma):
    """Gets value of n in a gaussian curve centered in 0 with sigma = sigma
    """
    return np.e**(-1/2 * (n / sigma)**2)

def get_rectangular_value(n,width):
    """Gets value of n in a rectangular function centered in 0 with width = width
    """
    return np.where(abs(n)<width,1,0)

def plot_color_wheel(ax, cmap):
        
    # Define colormap normalization for 0 to 2*pi
    norm_2 = colors.Normalize(0, 2*np.pi) 
        
    # Plot a color mesh on the polar plot
    # with the color set by the angle
    n = 360  #the number of secants for the mesh
    t = np.linspace(0,2*np.pi,n)   #theta values
    r = np.linspace(.6,1,2)        #radius values change 0.6 to 0 for full circle
    rg, tg = np.meshgrid(r,t)      #create a r,theta meshgrid
    c = tg                         #define color values as theta value
    ax.pcolormesh(t, r, c.T, norm=norm_2, cmap=cmap, shading='auto')  #plot the colormesh on axis with colormap
    ax.set(yticklabels=[],
            xticks=[0,np.pi/2,np.pi,np.pi*3/2],
            xticklabels=["E","N","W","S"])
    ax.tick_params(pad=4,labelsize=14)      #cosmetic changes to tick labels
    ax.spines['polar'].set_visible(False)
    ax.set_title('Head direction color wheel', y=1.0, pad=30)

class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()