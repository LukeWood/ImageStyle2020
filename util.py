import multiprocessing.pool
from functools import partial
import os


def _iter_valid_files(directory, white_list_formats, follow_links):
    """Iterates on files with extension in `white_list_formats` contained in `directory`.
    # Arguments
        directory: Absolute path to the directory
            containing files to be counted
        white_list_formats: Set of strings containing allowed extensions for
            the files to be counted.
        follow_links: Boolean.
    # Yields
        Tuple of (root, filename) with extension in `white_list_formats`.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links),
                      key=lambda x: x[0])

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    yield root, fname


def _count_valid_files_in_directory(directory,
                                    white_list_formats,
                                    follow_links):
    """Counts files with extension in `white_list_formats` contained in `directory`.
    # Arguments
        directory: absolute path to the directory
            containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        follow_links: boolean.
    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    num_files = len(list(
        _iter_valid_files(directory, white_list_formats, follow_links)))
    start, stop = 0, num_files
    return stop - start


def count_num_samples(directory):
    """
    From Keras DirectoryIterator
    """
    classes = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)

    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm'}
    pool = multiprocessing.pool.ThreadPool()
    function_partial = partial(_count_valid_files_in_directory,
                               white_list_formats=white_list_formats,
                               follow_links=False)
    num_samples = sum(pool.map(function_partial,
                               (os.path.join(directory, subdir)
                                for subdir in classes)))
    pool.close()
    pool.join()
    return num_samples
