"""Parallel computing utilities."""

from joblib import Parallel
from tqdm import tqdm


class ProgressParallel(Parallel):
    """Joblib Parallel extended to use ``tqdm`` progress bar.

    `This class was implemented by @user394430
    <https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib>`_.

    Parameters
    ----------
    use_tqdm : bool, optional
        Whether or not to print progress using tqdm. Default is ``True``.
    total : int, optional
        Length of the iterator. This argument is passed to ``tqdm``. Default is
        ``None``.
    tqdm_kwargs : dict, optional
        Kwargs arguments to pass to ``tqdm``. Default is ``None``.
    *args : iterable, optional
        Additional arguments passed to :class:`joblib.Parallel`.
    **kwargs : dict, optional
        Additional arguments passed to :class:`joblib.Parallel`.

    See Also
    --------
    joblib.Parallel : This class' superclass.
    """  # noqa: E501

    def __init__(self, use_tqdm=True, total=None, tqdm_kwargs=None, *args, **kwargs):
        if tqdm_kwargs is None:
            tqdm_kwargs = {}
        self._use_tqdm = use_tqdm
        self._total = total
        self._tqdm_kwargs = tqdm_kwargs
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(
            disable=not self._use_tqdm, total=self._total, **self._tqdm_kwargs
        ) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        """Control printing progress via the ``tqdm`` progress bar."""
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
