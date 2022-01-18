import matplotlib as plt

def get_cmap(n, name='hsv'):
        """
        Get a colormap instance with n numbers of entries.

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