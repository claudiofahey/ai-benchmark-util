"""Various commonly-used functions and classes.
   This only has dependencies on packages that are standard with Anaconda Python.
"""
from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import shutil
import os
import itertools
import datetime
import scipy.interpolate
import scipy.stats
import math
import six
import re
import bz2
import codecs
import glob
import json
import re
import dateutil
import tarfile
import io
import IPython.display
import ipywidgets
import copy
import uuid
import warnings
import traceback
import sys
import io
import csv
import logging
import operator
import scipy.constants
from matplotlib import gridspec
from IPython.display import display, HTML
from contextlib import closing
from sklearn.externals.joblib import Parallel, delayed
from collections import namedtuple
from scipy.stats import rv_discrete
from scipy.optimize import OptimizeResult, minimize_scalar

#
# Dataframe manipulation
#

def flatten_multiindex_columns(df):
    df = df.copy()
    df.columns = ['_'.join([str(x) for x in col]) for col in df.columns.values]
    return df

def series_to_df(s, series_name=None):
    df = pd.DataFrame(s)
    if series_name: df.columns = [series_name]
    return df

def add_cumsum(df, src_col, new_col):
    df[new_col] = df[src_col].cumsum()
    return df

def string_list_subtract_regex(string_list, exclude_regex_list):
    if exclude_regex_list:
        result = []
        exclude_regex = '|'.join(['%s$' % x for x in exclude_regex_list])
        compiled = re.compile(exclude_regex)
        result = [s for s in string_list if not compiled.match(s)]
        return result
    else:
        return string_list

def get_varying_column_names(df, exclude_cols=[], ignore_nan=False):
    """Return names of columns with more than one value.
    Column names that match regular expressions in exclude_cols will not be returned.
    If ignore_nan is True, column values with NaN are not considered different."""
    check_cols = string_list_subtract_regex(df.columns, exclude_cols)
    take_cols = []
    if len(df) > 1:
        for col in check_cols:
            significant_values = df[col]
            if ignore_nan:
                significant_values = significant_values[~significant_values.isnull()]
            try:
                value_counts = significant_values.apply(make_hash).value_counts()
                if len(value_counts) > 1:
                    take_cols.append(col)
            except:
                # If an error occurs, ignore the column.
                pass
    return take_cols

def take_varying_columns(df, exclude_cols=[], ignore_nan=False):
    take_cols = get_varying_column_names(df, exclude_cols=exclude_cols, ignore_nan=ignore_nan)
    return df[take_cols]

def get_common_column_names(df, exclude_cols=[], ignore_nan=False):
    """Return names of columns with one value. All columns are returned if there are 0 or 1 rows.
    Column names that match regular expressions in exclude_cols will not be returned.
    If ignore_nan is True, column values with NaN are not considered different."""
    check_cols = string_list_subtract_regex(df.columns, exclude_cols)
    take_cols = []
    if len(df) <= 1:
        take_cols = check_cols
    else:
        for col in check_cols:
            significant_values = df[col]
            if ignore_nan:
                significant_values = significant_values[~significant_values.isnull()]
            value_counts = significant_values.apply(make_hash).value_counts()
            if len(value_counts) <= 1:
                take_cols.append(col)
    return take_cols

def take_common_columns(df, exclude_cols=[], ignore_nan=False):
    take_cols = get_common_column_names(df, exclude_cols=exclude_cols, ignore_nan=ignore_nan)
    return df[take_cols]

def build_mask_string(row, df_name='d'):
    row_criteria = []
    for col in row.index.values:
        value = row[col]
        if isinstance(value, str):
            col_criteria = "({2}.{0}=='{1}')".format(col, value.replace("'", "\\'"), df_name)
        else:
            col_criteria = '({2}.{0}=={1})'.format(col, value, df_name)
        row_criteria.append(col_criteria)
    return '&\n'.join(row_criteria)

def dataframe_indexes_to_columns(df):
    """
    Adds index columns as regular columns if necessary. Modifies df.
    """
    for index_index, index_name in enumerate(df.index.names):
        if index_name not in df.columns:
            df[index_name] = df.index.get_level_values(index_name).values
    return df

def dict_list_to_dict(li, key):
    return dict(map(lambda x: (x[key], x), li))

def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    From: http://stackoverflow.com/questions/38987/how-can-i-merge-two-python-dictionaries-in-a-single-expression
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def format_group_name(columns_names, values):
    if isinstance(values, tuple):
        formatted_values = [str(int(x)) if type(x) is float else str(x) for x in values]
        return ",".join([str(x) for x in formatted_values])
    else:
        return str(values)

def format_dict(d, attr_sep=',', kv_sep='='):
    return attr_sep.join(['%s%s%s'%(k,kv_sep,v) for (k,v) in sorted(d.items())])

def flatten_list(li):
    """See also numpy.ravel()."""
    result = []
    for item in li:
        result.extend(item)
    return result

def force_float_or_nan(f):
    try:
        return np.float(f)
    except:
        return np.nan

def fix_column_types(df, dtypes):
    """Define datatypes for each column. Note that integers/bools will be imported as floats to support NaNs.
    Modifies passed dataframe in place."""

    # Fix all data types.
    for column, dtype in dtypes.items():
        if column not in df:
            df[column] = np.nan
        if dtype == 'datetime':
            df[column] = df[column].apply(lambda t: force_utc(t))
        elif dtype == 'float':
            df[column] = df[column].apply(lambda f: force_float_or_nan(f))
        elif dtype in ['str','unicode']:
            df[column] = df[column].fillna('').astype('unicode')
        elif dtype == 'ascii':
            df[column] = df[column].fillna('').astype('str')
        elif dtype == 'bool':
            # Keep NaNs but convert any other value too boolean
            mask = df[column].isnull()
            df.loc[~mask,column] = df.loc[~mask,column].astype(dtype)
        elif dtype == 'object':
            pass
        else:
            df[column] = df[column].astype(dtype)

    # Convert column names to consistent string type (unicode vs ascii)
    df.columns = map(str,df.columns.values)
    return df

def create_uuid_from_fields(rec, cols):
    """Create a deterministic uuid from specific record fields. This is used when there is not a recorded test_uuid value."""
    data = ''
    for col in cols:
        if col in rec and pd.notnull(rec[col]):
            value = str(rec[col])
            if value:
                data += value
                data += '|'
    if data == '':
        raise Exception('Unable to find any fields to create a deterministic UUID')
    gen_uuid = str(uuid.uuid5(uuid.UUID('035abefa-9a3a-402b-b573-d77327c7b532'), data))
    print('create_uuid_from_fields: Generated %s from %s' % (gen_uuid, data))
    return gen_uuid

def concat_dataframe_dicts(dataframe_dict_list):
    """Concatenate dataframes in matching keys in list of dicts."""
    result = {}
    keys = [set(d.keys()) for d in dataframe_dict_list if not d is None]
    if keys:
        keys = set.union(*keys)
    for df_name in keys:
        result[df_name] = pd.concat([d[df_name] for d in dataframe_dict_list if not d is None and df_name in d])
    return result

def filter_dataframe_mask(df, warn_on_no_match=True, **kwargs):
    """Return a mask to filter a dataframe based on keyword arguments.
    Values that are tuples will be applied with between().
    Lists and scalars will be applied with isin()."""
    mask = np.ones(len(df)).astype(bool)
    for col, criteria in kwargs.items():
        if isinstance(criteria,tuple):
            mask = mask & df[col].between(*criteria)
        else:
            if not isinstance(criteria,list):
                criteria = [criteria]
            mask = mask & df[col].isin(criteria)
        if warn_on_no_match and np.sum(mask) == 0:
            print('filter_dataframe: No matching records after filtering on %s=%s' % (col, criteria), file=sys.stderr)
            warn_on_no_match = False
    return mask

def filter_dataframe(df, warn_on_no_match=True, **kwargs):
    """Filter a dataframe based on keyword arguments.
    Values that are tuples will be applied with between().
    Lists and scalars will be applied with isin()."""
    mask = filter_dataframe_mask(df, warn_on_no_match=warn_on_no_match, **kwargs)
    return df[mask]

def reorder_index_columns(df, begin_cols=[], end_cols=[]):
    """Change the left-right order of columns in a multiindex."""
    begin_cols = [c for c in begin_cols if c in df.index.names]
    end_cols = [c for c in end_cols if c in df.index.names]
    index_cols = begin_cols + sorted(list(set(df.index.names) - set(begin_cols) - set(end_cols))) + end_cols
    return df.reset_index().set_index(index_cols).sort()

def reorder_dataframe_columns(df, begin_cols=[], end_cols=[], sort_middle=False):
    """Change the order of columns in a dataframe."""
    all_cols = list(df.columns)
    begin_cols = [c for c in begin_cols if c in all_cols]
    end_cols = [c for c in end_cols if c in all_cols]
    middle_cols = [c for c in all_cols if c not in begin_cols and c not in end_cols]
    if sort_middle:
        middle_cols = sorted(middle_cols)    
    ordered_cols = begin_cols + middle_cols + end_cols
    return df[ordered_cols]

def percentile(n):
    """Based on http://stackoverflow.com/questions/17578115/pass-percentiles-to-pandas-agg-function."""
    def _percentile(x):
        return np.percentile(x, n)
    _percentile.__name__ = 'percentile_%s' % n
    return _percentile

def rename_columns(df, column_rename_dict):
    """Rename columns. Modifies passed dataframe in place.
    OBSOLETE: Use rename_dataframe_column()"""
    for old_column, new_column in column_rename_dict.items():
        if old_column in df:
            if not new_column is None:
                if new_column in df:
                    mask = (df[new_column].isnull()) & (~df[old_column].isnull())
                    df.loc[mask,new_column] = df.loc[mask,old_column]
                else:
                    df[new_column] = df[old_column]
            del df[old_column]
    return df

def rename_dataframe_column(df, columns, level=0):
    """Rename the names of a column.
    From http://stackoverflow.com/questions/29369568/python-pandas-rename-single-column-label-in-multi-index-dataframe"""
    def rename_apply (x, rename_dict):
        try:
            return rename_dict[x]
        except KeyError:
            return x
        
    if  isinstance(df.columns, pd.core.index.MultiIndex):
        df.columns = df.columns.set_levels([rename_apply(x, rename_dict = columns ) for x in df.columns.levels[level]], level= level)
    else:
        df.columns =                       [rename_apply(x, rename_dict = columns ) for x in df.columns              ] 
    return df

def column_name_to_title(s):    
    s = s.replace('_',' ')
    return ' '.join([w[0].upper() + w[1:] for w in s.split(' ')])

def crossjoin_df(df1, df2, multi_index=False, **kwargs):
    """
    Make a cross join (cartesian product) between two dataframes by using a constant temporary key.
    Also sets a MultiIndex which is the cartesian product of the indices of the input dataframes.
    See: https://github.com/pydata/pandas/issues/5401
    :param df1 dataframe 1
    :param df1 dataframe 2
    :param kwargs keyword arguments that will be passed to pd.merge()
    :return cross join of df1 and df2
    """
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    if multi_index:
        res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res

def report_diff_html(x, equal_nan=False):
    """
    From http://stackoverflow.com/questions/17095101/outputting-difference-in-two-pandas-dataframes-side-by-side-highlighting-the-d
    """
    if x[0]==x[1]:
        return unicode(x[0].__str__())
    elif pd.isnull(x[0]) and pd.isnull(x[1]):
        if equal_nan:
            return u'nan'
        return u'<table style="background-color:#00ff00;font-weight:bold;">'+\
            '<tr><td>%s</td></tr><tr><td>%s</td></tr></table>' % ('nan', 'nan')
    elif pd.isnull(x[0]) and ~pd.isnull(x[1]):
        return u'<table style="background-color:#ffff00;font-weight:bold;">'+\
            '<tr><td>%s</td></tr><tr><td>%s</td></tr></table>' % ('nan', x[1])
    elif ~pd.isnull(x[0]) and pd.isnull(x[1]):
        return u'<table style="background-color:#0000ff;font-weight:bold;">'+\
            '<tr><td>%s</td></tr><tr><td>%s</td></tr></table>' % (x[0],'nan')
    else:
        return u'<table style="background-color:#ff0000;font-weight:bold;">'+\
            '<tr><td>%s</td></tr><tr><td>%s</td></tr></table>' % (x[0], x[1])

def compare_dataframes_html(df1, df2, **kwargs):
    """
    From http://stackoverflow.com/questions/17095101/outputting-difference-in-two-pandas-dataframes-side-by-side-highlighting-the-d
    """
    panel = pd.Panel(dict(df1=df1, df2=df2))
    if pd.options.display.max_colwidth < 500:
        pd.options.display.max_colwidth = 500  # You need this, otherwise pandas will limit your HTML strings to 50 characters
    return HTML(panel.apply(lambda x: report_diff_html(x, **kwargs), axis=0).to_html(escape=False))

#
# File I/O
#

def move_data_files(src, dst):
    if not os.path.isdir(dst):
        os.mkdir(dst)
    for filename in glob.glob(src):
        print('Moving ' + filename + ' -> ' + dst)
        shutil.move(filename, dst)

def text_files_iterator(src, verbose=False):
    """Iterate over lines in text files. src can contain wildcards. Files may be compressed with bzip2."""
    for filename in glob.glob(src):
        if verbose: print('Reading %s ' % filename)
        ext = os.path.splitext(filename)[1]
        if ext == '.bz2':
            with closing(bz2.BZ2File(filename, 'rb')) as data_file:
                reader = codecs.getreader("utf-8")
                for line in reader(data_file):
                    yield line
        else:
            with open(filename, 'r') as file:
                for line in file:
                    yield line

def load_json_from_file(filename):
    ext = os.path.splitext(filename)[1]
    if ext == '.bz2':
        with closing(bz2.BZ2File(filename, 'rb')) as data_file:
            reader = codecs.getreader("utf-8")
            data = json.load(reader(data_file))
    else:
        with open(filename) as data_file:
            data = json.load(data_file)
    return data

def glob_file_list(filespecs):
    if not isinstance(filespecs,list):
        filespecs = [filespecs]
    return sum(map(glob.glob, filespecs), [])

def load_json_records(src, verbose=False, n_jobs=-1):
    recs = []
    filenames = glob_file_list(src)
    print('Loading records from %d files...' % len(filenames))
    pjobs = [delayed(load_json_from_file)(filename) for filename in filenames]
    file_record_list = Parallel(n_jobs=n_jobs)(pjobs)       # list of lists of records
    recs = flatten_list(file_record_list)
    return recs

def load_json_from_file_as_dataframe(filename, filename_column_name='loaded_filename', ignore_error=False):
    try:
        df = pd.DataFrame(load_json_from_file(filename))
        df[filename_column_name] = filename
        return df
    except Exception:
        print('EXCEPTION while loading JSON file %s: %s' % (filename, traceback.format_exc()), file=sys.stderr)
        if ignore_error:
            return pd.DataFrame()
        else:
            raise

def load_json_records_as_dataframe(src, verbose=False, n_jobs=-1, ignore_error=False):
    recs = []
    filenames = glob_file_list(src)
    print('Loading records from %d files...' % len(filenames))
    pjobs = [delayed(load_json_from_file_as_dataframe)(filename, ignore_error=ignore_error) for filename in filenames]
    df_list = Parallel(n_jobs=n_jobs)(pjobs)
    return pd.concat(df_list, ignore_index=True, copy=False, sort=False)

def save_json_to_file(data, filename, sort_keys=False, indent=None, ensure_ascii=False):
    ext = os.path.splitext(filename)[1]
    temp_file_name = '%s.tmp%s' % os.path.splitext(filename)
    if ext == '.bz2':
        with closing(bz2.BZ2File(temp_file_name, 'wb')) as data_file:
            json.dump(data, data_file, sort_keys=sort_keys, indent=indent, ensure_ascii=ensure_ascii)
    else:
        with open(temp_file_name, 'w') as data_file:
            json.dump(data, data_file, sort_keys=sort_keys, indent=indent, ensure_ascii=ensure_ascii)
    os.rename(temp_file_name, filename)

#
# Plotting
#

def plot_groups(df, x_col, y_col, group_by_columns=None, title=None, xlabel=None, ylabel=None, max_legend_items=10,
                sort_columns=[0], semilogx=False, semilogy=False, agg=None, xlim=None, ylim=None, xticks=None):
    # Cyclers for different plot styles
    lines = ['-','--','-.',':']
    markers = ['o','s','*','x','D','v','^','<','>','8','p','|']
    colors = ['r','b','g','c','m','y','k']
    lineCycler = itertools.cycle(lines)
    markerCycler = itertools.cycle(markers)
    colorCycler = itertools.cycle(colors)

    fig = plt.figure(title, figsize=(20,10))
    fig.clf()
    if title:
        fig.suptitle(title, fontsize=12)
    num_groups = 0

    if group_by_columns is None:
        num_groups = 1
        df.plot(x=x_col, y=y_col, style=next(markerCycler) + next(colorCycler) + next(lineCycler))
    else:
        group_by_columns = list(set(group_by_columns) - {x_col})
        if sort_columns == [0]:
            sort_columns = [x_col]
        for name, group in df.groupby(group_by_columns):
            num_groups += 1
            nameStr = format_group_name(group_by_columns, name)
            if len(group) > 100:
                style = next(colorCycler) + next(lineCycler)
            else:
                style = next(markerCycler) + next(colorCycler) + next(lineCycler)
            if sort_columns is None:
                sorted_group = group
            else:
                sorted_group = group.sort_values(sort_columns)
            plt.plot(sorted_group[x_col].values, sorted_group[y_col].values, style, label=nameStr)

    if agg is not None:
        agg_df = df.groupby(by=[x_col], as_index=False).agg({y_col: agg})
        plt.plot(agg_df[x_col].values, agg_df[y_col].values, 'xk-', label=agg)

    axes = plt.gca()
    if xlabel is None:
        xlabel = x_col
    if ylabel is None:
        ylabel = y_col
    if num_groups <= max_legend_items:
        axes.legend(loc='best')
    else:
        print('plot_groups: not showing legend because num_groups=%d' % num_groups)
    if semilogx:
        axes.semilogx()
        fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        axes.xaxis.set_major_formatter(fmt)
    if semilogy:
        axes.semilogy()
        fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        axes.yaxis.set_major_formatter(fmt)
    if xlim:
        axes.set_xlim(xlim)
    if ylim:
        axes.set_ylim(ylim)
    if xticks:
        axes.xaxis.set_ticks(xticks)
    fig.autofmt_xdate()
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    #plt.show()
    return fig

def show_dataframe_mpl(df, axes, col_labels=True):
    """See also pd.DataFrame.plot(table=True)."""
    cellText = list(df.values)
    if col_labels:
        colLabels=list(df.columns.values)
    else:
        colLabels = None
    tbl = axes.table(cellText=cellText, colLabels=colLabels, loc='center')
    axes.axis('off')
#     tbl.set_fontsize(5)
#     for cell in tbl.get_child_artists():
#         cell.set_height(0.04)
#         cell.set_linewidth(0.001)

def show_series_mpl(series, axes):
    # df = pd.DataFrame(series).reset_index()
    df = pd.DataFrame({'index': series.index.values, 'value': series.values})
    return show_dataframe_mpl(df, axes, col_labels=False)

def expand_xlim(x, axes, margin=0.0):
    """If necessary, expand x limit to that x-margin and x+margin are visible."""
    a, b = axes.get_xlim()
    a = min(a, x - margin)
    b = max(b, x + margin)
    axes.set_xlim(a, b)

def expand_ylim(y, axes, margin=0.0):
    """If necessary, expand y limit to that y-margin and y+margin are visible."""
    a, b = axes.get_ylim()
    a = min(a, y - margin)
    b = max(b, y + margin)
    axes.set_ylim(a, b)

#
# Optimization
#

class CachedFunction(object):
    """Caches function calls with the same arguments."""

    def __init__(self, fun, record_history=False):
        self.fun = fun
        self.cached_points = {}
        self.record_history = record_history
        self.history = []       # ordered history of uncached function evaluations
        self.uncached_fev = 0   # number of actual uncached function evaluations (cache misses)
        self.cached_fev = 0     # number of cached function calls (cache hits)

    def __call__(self, *args, **kwargs):
        cache_key = make_hashable((args, kwargs))
        # logging.info('cache_key=%s' % str(cache_key))
        try:
            y = self.cached_points[cache_key]
            self.cached_fev += 1
            return y
        except KeyError:
            # logging.info('Calling function to evaluate cache_key=%s' % str(cache_key))
            self.uncached_fev += 1
            y = self.fun(*args, **kwargs)
            self.cached_points[cache_key] = y
            if self.record_history:
                self.history.append(args + (kwargs, y,))
            return y

class SmoothedDiscreteFunction(object):
    """Smoothes a scalar function of a single discrete variable by linear interpolation between points."""

    def __init__(self, fun, x_domain):
        """
        Args:
            x_domain (np.ndarray): Array of values that represent the discrete domain of the function.
                Values can have type int or float.
        """
        self.fun = fun
        self.x_domain = np.sort(x_domain)

    def __call__(self, x):
        if x < self.x_domain[0] or x > self.x_domain[-1]:
            raise ValueError('x=%s is outside the domain [%s,%s]' % (x, self.x_domain[0], self.x_domain[-1]))
        x0_index = np.searchsorted(self.x_domain, x, side='right') - 1
        if self.x_domain[x0_index] == x:
            y = self.fun(x)
            logging.info('SmoothedDiscreteFunction(%f) = fun(%f) = %f' % (x, x, y))
            return y
        X = self.x_domain[x0_index:x0_index+2]
        Y = np.array([self.fun(xx) for xx in X])
        ifun = scipy.interpolate.interp1d(X, Y, assume_sorted=True, copy=False)
        y = ifun([x])[0]
        logging.info('SmoothedDiscreteFunction(%f) ~ fun(%s) = %f' % (x, X, y))
        return y

class SteppedDiscreteFunction(object):
    """Provided with a scalar function of multiple discrete variables, this will extend the domain
    to all real numbers by rounding down to the nearest value in the domain. This is performed for each
    dimension separately. This will create multi-dimensional "step" functions that are flat (zero gradient)
    except at the points in the original domain, where the gradients may be undefined.
    This can be used with `CachedFunction` to round down to the nearest point and cache that point."""

    def __init__(self, fun, x_domain):
        """
        Args:
            x_domain (list(np.ndarray)): Array of values that represent the discrete domain of the function.
                Values can have type int or float.
        """
        self.fun = fun
        self.x_domain = [np.sort(xi_domain) for xi_domain in x_domain]

    def convert_x(self, x):
        x = np.atleast_1d(x)
        assert(len(x) == len(self.x_domain))
        x_nearest = np.zeros(len(self.x_domain))
        for i in range(len(self.x_domain)):
            if x[i] <= self.x_domain[i][0]:
                x_nearest[i] = self.x_domain[i][0]
            elif x[i] >= self.x_domain[i][-1]:
                x_nearest[i] = self.x_domain[i][-1]
            else:
                xi0_index = np.searchsorted(self.x_domain[i], x[i], side='right') - 1
                x_nearest[i] = self.x_domain[i][xi0_index]
        return x_nearest

    def __call__(self, x):
        x_nearest = self.convert_x(x)
        y = self.fun(x_nearest)
        # logging.info('SteppedDiscreteFunction(%s) ~ fun(%s) = %f' % (x, x_nearest, y))
        return y

class PandasSeriesFunction(object):
    """Make a function out of a Pandas Series object."""
    def __init__(self, series):
        self.series = series

    def __call__(self, x):
        return self.series.ix[tuple(np.atleast_1d(x))]

class LoggingFunction(object):
    """This function wrapper will log all function calls."""
    def __init__(self, fun=None, name=None):
        self.fun = fun
        if name is None:
            try:
                name = fun.__name__
            except:
                name = 'LoggingFunction'
        self.name = name

    def __call__(self, *args, **kwargs):
        arg_str = [repr(a) for a in args]
        kwarg_str = ['%s=%s' % (k,repr(v)) for k,v in kwargs.iteritems()]
        both_str = arg_str + kwarg_str
        joined_str = ', '.join(both_str)
        if self.fun is None:
            logging.info('%s(%s)' % (self.name, joined_str))
        else:
            result = self.fun(*args, **kwargs)
            logging.info('%s(%s) -> %s' % (self.name, joined_str, result))
            return result

class defaultlist(list):
    """Based on http://stackoverflow.com/questions/869778/populating-a-list-array-by-index-in-python."""
    def __init__(self, iterable=None, default_factory=None):
        args = []
        if iterable:
            args = [iterable]
        super(defaultlist, self).__init__(*args)
        if default_factory is None:
            default_factory = lambda: None
        self.default_factory = default_factory

    def __setitem__(self, index, value):
        size = len(self)
        if index >= size:
            self.extend(self.default_factory() for _ in range(size, index + 1))
        list.__setitem__(self, index, value)

class ArgsToArrayMap(object):
    def __init__(self, arg_map):
        """
        Args:
            arg_map (list): List of dict with the following keys:
              kwarg_name: Name of keyword argument.
              arg_number: Position of argument. Must use exactly one of kwarg or arg_number.
              array_size: If argument should be a list, specify the size.
              value_if_missing: If present, this value will be used by args_to_array if the argument is missing.
                If this key is not present, a missing argument will produce an AttributeError exception.
              fixed_arg_value: If present, this argument will not be included in the array but array_to_args will include
                this fixed value.
              dtype: If present, arguments returned by array_to_args() will be converted to this type by
                passing to this function as a parameter.
        """
        self.arg_map = arg_map

    def args_to_array(self, *args, **kwargs):
        """
        Returns the array as a list.
        """
        a = []
        for arg_info in self.arg_map:
            if 'fixed_arg_value' not in arg_info:
                kwarg_name = arg_info.get('kwarg_name')            
                if kwarg_name is not None:
                    if 'value_if_missing' in arg_info:
                        arg_value = kwargs.get(kwarg_name, arg_info['value_if_missing'])
                    else:
                        arg_value = kwargs[kwarg_name]
                else:
                    arg_number = arg_info.get('arg_number')
                    if arg_number is not None:
                        if arg_number >= len(args) and 'value_if_missing' in arg_info:
                            arg_value = arg_info['value_if_missing']
                        else:
                            arg_value = args[arg_number]
                    else:
                        raise AttributeError('You must specify kwarg_name or arg_number.')
                array_size = arg_info.get('array_size')
                if array_size is None:
                    arg_value = [arg_value]
                elif len(arg_value) != array_size:
                    raise ValueError('Value for argument %s has incorrect size.' % str(arg_info))
                a.extend(arg_value)
        return a

    def array_to_args(self, array):
        args = defaultlist()
        kwargs = {}
        a = list(array)
        for arg_info in self.arg_map:
            if 'fixed_arg_value' in arg_info:
                arg_value = arg_info['fixed_arg_value']
            else:
                num_elements = arg_info.get('array_size', 1)
                arg_value = a[0:num_elements]
                a = a[num_elements:]
                if arg_info.get('array_size') is None:
                    arg_value = arg_value[0]
            dtype = arg_info.get('dtype')
            if dtype is not None:
                arg_value = dtype(arg_value)
            if arg_info.get('kwarg_name') is not None:                
                kwargs[arg_info.get('kwarg_name')] = arg_value
            elif arg_info.get('arg_number') is not None:
                args[arg_info.get('arg_number')] = arg_value
        return args, kwargs

class AttributesToArray(object):
    """Maps an array to/from object attributes.

    >>> def __init__(self):
    >>>     arg_map = [
    >>>         dict(kwarg_name='trend_parameters', array_size=self.num_trend_parameters, dtype=np.array),
    >>>         dict(kwarg_name='decay_parameters', array_size=self.num_decay_parameters, dtype=np.array),
    >>>         dict(kwarg_name='season_parameters', array_size=self.num_season_parameters, dtype=np.array),
    >>>         ]
    >>>     args_to_array_map = ArgsToArrayMap(arg_map)
    >>>     self.attributes_to_array = AttributesToArray(args_to_array_map, self)

    """
    def __init__(self, args_to_array_map, bound_object):
        self.bound_object = bound_object
        self.args_to_array_map = args_to_array_map

    def set_array(self, array):
        args, kwargs = self.args_to_array_map.array_to_args(array)
        for k,v in kwargs.iteritems():
            self.bound_object.__dict__[k] = v

    def get_array(self):
        array = self.args_to_array_map.args_to_array(**self.bound_object.__dict__)
        return array

class ArgsToArrayFunction(object):
    """This function wrapper will convert a function called with positional and 
    keyword arguments to a function called with an array.
    This is useful when passing to minimize()."""
    def __init__(self, fun, args_to_array_map):
        """
        Args:
            args_to_array_map(ArgsToArrayMap):
        """
        self.fun = fun
        self.args_to_array_map = args_to_array_map

    def __call__(self, array, *extra_args, **override_kwargs):
        """Convert call with array to call with args and kwargs."""
        args, kwargs = self.args_to_array_map.array_to_args(array)
        kwargs.update(override_kwargs)
        args += extra_args
        result = self.fun(*args, **kwargs)
        return result

def fit_parabola(X, Y):
    if not (len(X) == 3 and len(Y) == 3):
        raise ValueError()
    M = np.matrix(np.array([X**2, X, np.ones(3)]).T)
    a,b,c = np.linalg.solve(M,Y) # coefficients of ax**2 + bx + c
    return a,b,c

def find_vertex_x_of_positive_parabola(X, Y):
    a,b,c = fit_parabola(X,Y)
    if a <= 0:
        raise ValueError('Parabola not positive')
    min_x = -b / (2.0*a)
    return min_x

def discrete_scalar_convex_minimizer(fun, x0, x_domain, args=(), maxfev=None, maxiter=100, callback=None, **options):
    """Minimize a scalar function of a single variable that takes on a finite number of values.
    The function must be "non-strictly" convex, meaning that it is possible for f(a) == f(b) around the minimum.
    This trivial optimization approach currently begins with the first x in x_domain and moves to
    the subsequent x until the function starts increasing.
    This function is NOT recommended. Use `scalar_near_convex_minimizer` instead.
    """
    bestx_index = 0
    bestx = x_domain[bestx_index]
    besty = fun(bestx)
    funcalls = 1
    niter = 0
    improved = True
    stop = False

    while improved and not stop and niter < maxiter:
        niter += 1
        testx_index = bestx_index + 1
        testx = x_domain[testx_index]
        testy = fun(testx, *args)
        funcalls += 1
        if testy <= besty:
            # Still going downhill or flat
            bestx_index = testx_index
            bestx = testx
            besty = testy
        else: #if testy > besty:
            # We have now started going up
            stop = True
        if callback is not None:
            callback(bestx)
        if maxfev is not None and funcalls >= maxfev:
            stop = True
            break

    return OptimizeResult(fun=besty, x=bestx, nit=niter, nfev=funcalls, success=(niter > 1))

def scalar_gap_filling_minimizer(fun, bracket, args=(), tol=1e-6, maxfev=None, maxiter=100, callback=None, verbose=False,
        parabolic_method=False, golden_section_method=False, **options):
    """Find a local minimum of a scalar function of a single variable.
    The function may have flat spots where f(a) == f(b) for a != b and this method will
    attempt to search around and within the flat spots.
    The function must have exactly one local minimum in the bracket.
    This method maintains a left and right bracket, where the function value is greater than the best known minimum.
    It also maintains a list of best x values, and the function values at all of these x values equals the best known minimum.
    At each iteration, it finds the largest gap in these x values (including the brackets) and selects
    the point in the center of the largest gap.
    It will then either adjust the bracket or add to the list of best x values.
    The method terminates when the largest gap is less than bracket_tol.

    Args:
        bracket (tuple): A tuple of the bounds of the function (x_min, x_max).
            Optionally, a middle point can be specified and it will be the initial best point.
        tol (float): The method terminates when the largest gap is less than this value.

    """
    # bestx is a list.
    # besty is a scalar and equals f(x) for all x in bestx.

    funcalls = 0

    # print('parabolic_method=%s,golden_section_method=%s' % (parabolic_method,golden_section_method))

    if len(bracket) == 2:
        bracket_left_x = bracket[0]
        bracket_right_x = bracket[1]
        bestx = [np.mean([bracket_left_x, bracket_right_x])]
        a = bracket_left_x
        b = bracket_right_x
        if golden_section_method:
            bestx = [b - (b - a) / scipy.constants.golden]
        else:
            bestx = [np.mean([a, b])]
    elif len(bracket) == 3:
        bracket_left_x = bracket[0]
        bracket_right_x = bracket[2]
        bestx = [bracket[1]]
    else:
        raise ValueError('Invalid bracket')

    if not (bracket_left_x <= bestx[0] <= bracket_right_x):
        raise ValueError('Invalid bracket')

    # Evaluate function at bestx.
    besty = fun(bestx[0])
    funcalls += 1

    # Evaluate function at brackets to determine if they are better than the initial bestx.
    bracket_left_y = fun(bracket_left_x, *args)
    bracket_right_y = fun(bracket_right_x, *args)
    funcalls += 2
    if bracket_left_y < besty:
        bestx = [bracket_left_x]
        besty = bracket_left_y
    if bracket_right_y < besty:
        bestx = [bracket_right_x]
        besty = bracket_right_y

    if verbose: logging.info('bracket=(%f,%s,%f); besty=%f' % (bracket_left_x, str(bestx), bracket_right_x, besty))

    niter = 0
    while niter < maxiter:
        niter += 1

        X = np.array([bracket_left_x] +  bestx               + [bracket_right_x])
        Y = np.array([bracket_left_y] + [besty] * len(bestx) + [bracket_right_y])
        testx = None
        testx_index = None

        # If we have exactly one bestx, then fit a parabola to the 3 points and test the vertex.
        if parabolic_method and len(bestx) == 1:
            if verbose: logging.info('Attempting parabolic method')
            try:
                # Attempt to fit a parabola to the 3 points and find the vertex.
                testx = find_vertex_x_of_positive_parabola(X, Y)
                if verbose: logging.info('Parabolic method returned testx=%f' % testx)
                if testx <= bracket_left_x or testx >= bracket_right_x or testx == bestx[0]:
                    testx = None
                elif testx <= bestx[0]:
                    testx_index = 0
                else:
                    testx_index = 1
            except:
                # This will happen if a parabola can't be fit through the 3 points.
                # Ignore error and use the gap method below.
                testx = None
        if testx is None:
            # Measure gaps in brackets and bestx and find the largest one.
            if verbose: logging.info('Attempting gap method')
            gaps = np.diff(X)
            testx_index = np.argmax(gaps)
            gapsize = gaps[testx_index]
            if gapsize < tol:
                if verbose: logging.info('Achieved gap size tol')
                break

            # Pick a point between the largest gap.
            a = X[testx_index]
            b = X[testx_index + 1]
            if golden_section_method:
                golden_distance = (b - a) / scipy.constants.golden
                if bool(np.random.randint(low=0, high=2)):
                    testx = b - golden_distance
                else:
                    testx = a + golden_distance
            else:
                testx = np.mean([a, b])

            if verbose: logging.info('gapsize=%f, len(bestx)=%d, testx=%f' % (gapsize, len(bestx), testx))

        assert(testx is not None)
        assert(testx_index is not None)
        assert(bracket_left_x <= testx <= bracket_right_x)

        testy = fun(testx, *args)
        funcalls += 1

        add_to_bestx = False

        if testy < besty:
            # Found a point better than all others so far.
            # The new bracket will be the points to the immediate left and right of the test point.
            bestx = [testx]
            besty = testy
            bracket_left_x = X[testx_index]
            bracket_left_y = Y[testx_index]
            bracket_right_x = X[testx_index + 1]
            bracket_right_y = Y[testx_index + 1]
        elif testy > besty:
            # Point is worse than best. Reduce bracket.
            if testx_index == 0:
                # Test point was adjacent to left bracket.
                bracket_left_x = testx
                bracket_left_y = testy
            elif testx_index == len(X) - 2:
                # Test point was adjacent to right bracket.
                bracket_right_x = testx
                bracket_right_y = testy
            else:
                # Test point was inside the set of bestx points but is worse than besty.
                # This indicates more than one local minima or a round off error.
                # We will assume a round off error and handle it as if it had the same besty.
                add_to_bestx = True
        else:
            # Point is same as best. Add it to the bestx list.
            add_to_bestx = True

        if add_to_bestx:
            bestx = sorted(bestx + [testx])

        if verbose: logging.info('bracket=(%f,%s,%f); besty=%f' % (bracket_left_x, str(bestx), bracket_right_x, besty))
        if callback is not None:
            callback(bestx)
        if maxfev is not None and funcalls >= maxfev:
            break

    # Return the x that is in the median of bestx.
    bestx = bestx[int((len(bestx)-1)/2)]

    return OptimizeResult(fun=besty, x=bestx, nit=niter, nfev=funcalls, success=(niter > 1))

def multivariate_gap_filling_minimizer(fun, x0, bounds, args=(), tol=1e-6, maxfev=None, maxiter=2, callback=None, verbose=False, scalar_options={}, **options):
    """It is assumed that there is exactly one local minimum in the domain.
    This multivariate method uses `scalar_gap_filling_minimizer` repeatedly along each dimension
    for a fixed number of iterations. There is currently no other stopping criteria.

    TODO: Use Powell's method to optimize a linear combination of dimensions at a time.

    Args:
        x0 (np.ndarray): Initial guess.
        bounds: (min, max) pairs for each element in x, defining the bounds on that parameter.
        tol: See `scalar_near_convex_minimizer`.
    """

    ndims = len(x0)
    if len(bounds) != ndims:
        raise ValueError()

    bestx = x0
    besty = np.inf
    niter = 0
    funcalls = 0

    while niter < maxiter:
        niter += 1
        for i in range(ndims):
            # if verbose:
            #     logging.info('multivariate_near_convex_minimizer: dimension %d' % (i,))

            # Function of single variable that we will optimize during this iteration.
            def scalar_fun(x):
                testx = bestx
                testx[i] = x
                return fun(testx)

            bracket = (bounds[i][0], bestx[i], bounds[i][1])
            optresult = minimize_scalar(scalar_fun, bracket=bracket, tol=tol, method=scalar_gap_filling_minimizer, options=scalar_options)
            # if verbose:
            #     logging.info('minimize_scalar returned x=%f, y=%f' % (optresult.x, optresult.fun))
            bestx[i] = optresult.x
            besty = optresult.fun
            if verbose:
                logging.info('multivariate_gap_filling_minimizer: niter=%d, dim=%d, best f(%s) = %f' % (niter, i, str(bestx), besty))
            funcalls += optresult.nfev
        # if verbose:
        #     logging.info('multivariate_near_convex_minimizer: niter=%d, best f(%s) = %f' % (niter, str(bestx), besty))
        if maxfev is not None and funcalls >= maxfev:
            break

    return OptimizeResult(fun=besty, x=bestx, nit=niter, nfev=funcalls, success=(niter > 1))

def global_minimizer_spark(fun, x0=None, kwargs_list=[], kwargs_default={}, sc=None, callback=None,
        final_eval_kwargs=None,  return_all_as_list=False, return_all_as_dataframe=False, **options):
    """Minimize an arbitrary function by evaluating all possible points in parallel using Spark.

    Args:
        fun: A function that takes kwargs as input and outputs y. If y is a tuple, the first element must be the value
            to minimize. Subsequent values, if any, will be returned with the result allowing extra information to be returned.
            The value to minimize can be any comparable data type.
        kwargs_list (list(dict)): List of x points at which to evaluate function.
            Elements are dicts that represent kwargs to pass to the function.
        kwargs_default (dict): Optional parameters to pass to each function call.
        sc (SparkContext): The SparkContext to parallelize the function evaluations.
        return_all_evals (bool): If True, also returns all function evaluations as a list of (y, x) tuples.
            This will be returned in the fun_eval_list parameter.
        final_eval_kwargs (dict): If specified, re-evaluate the function at the minimum point but with this
            additional set of kwargs.
        x0: Not used but needed as a placeholder when called by scipy.optimize.minimize.
    """

    # Create RDD of function parameters.
    # We maintain two sets of parameters. 
    # kwargs_point varies between points.
    # kwargs_full includes kwarg_point and adds kwargs_default.
    params = []
    for kwargs_point in kwargs_list:
        kwargs_full = kwargs_default.copy()
        kwargs_full.update(kwargs_point)
        params.append((kwargs_point, kwargs_full))        
    params_rdd = sc.parallelize(params)

    # Evaluate function at each point in parallel using Spark.
    fun_eval_rdd = params_rdd.map(lambda param: (fun(**param[1]), param[0]))

    # Find the minimum y value. Secondary sort on the x value for tie breaking.
    best_y, best_x_kwargs = fun_eval_rdd.min(lambda x: (x[0][0],x[1]) if isinstance(x[0],tuple) else (x[0],x[1]))

    result = OptimizeResult(x=best_x_kwargs, nfev=len(params), success=True)

    if return_all_as_list:
        fun_eval_list = fun_eval_rdd.collect()
        result['fun_eval_list'] = fun_eval_list

    if return_all_as_dataframe:
        fun_eval_list = fun_eval_rdd.collect()
        df = pd.DataFrame([x for y,x in fun_eval_list])
        df['fun'] = [y[0] for y,x in fun_eval_list]
        result['df'] = df

    if final_eval_kwargs:
        kwargs_full = kwargs_default.copy()
        kwargs_full.update(best_x_kwargs)
        kwargs_full.update(final_eval_kwargs)
        best_y = fun(**kwargs_full)

    result['fun'] = best_y

    return result

#
# Misc functions
#

def regex_groups(regex, s, return_on_no_match=None, flags=0, search=False):
    if isinstance(s, six.string_types):
        if search:
            m = re.search(regex, s, flags=flags)
        else:
            m = re.match(regex, s, flags=flags)
        if m:
            return m.groups()
    return return_on_no_match

def regex_first_group(regex, s, return_on_no_match=None, flags=0, search=False):
    g = regex_groups(regex, s, return_on_no_match=[return_on_no_match], flags=flags, search=search)
    return g[0]

def print_arg(x):
    print(x)
    return x

def make_hash(o):
    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries).
    Based on http://stackoverflow.com/questions/5884066/hashing-a-python-dictionary
    """
    return hash(make_hashable(o))

def make_hashable(o):
    """
    Makes a hashable object from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries).
    Based on http://stackoverflow.com/questions/5884066/hashing-a-python-dictionary
    """
    if isinstance(o, (set, tuple, list, np.ndarray)):
        return tuple([make_hashable(e) for e in o])

    try:
        if np.isnan(o):
            # This prevents np.nan to have same hash as 0 or False.
            return '1b3c4484-61dd-4623-b60a-a4789eacecbd'
    except:
        pass

    try:
        if pd.isnull(o):
            # This prevents None to have same hash as 0 or False. Also ensures that hash(None) is consistent.
            return 'c3fe5446-ec19-4ec8-807a-8e739f9fc6b6'
    except:
        pass

    if isinstance(o, dict):
        new_o = copy.deepcopy(o)
        for k, v in new_o.items():
            new_o[k] = make_hashable(v)
        return tuple(frozenset(sorted(new_o.items())))
    else:
        return o

def force_utc(t):
    if isinstance(t,pd.Timestamp):
        if t.tz:
            result = t.tz_convert('UTC')
        else:
            result =  t.tz_localize('UTC')
    else:
        result = pd.to_datetime(t, utc=True)
    if type(result) != pd.tslib.Timestamp and type(result) != pd.tslib.NaTType:
        warnings.warn('force_utc: Conversion from %s resulted in type %s' % (type(t), type(result)))
        assert False
    return result

def linear_transform(x, x0, x1, y0, y1):
    return (x - x0) * (y1 - y0) / (x1 - x0) + y0

def linear_fit_to_2_points(point0, point1):
    """Return a linear estimator that passes through the points."""
    x0,y0 = point0
    x1,y1 = point1
    return lambda x: (x - x0) * (y1 - y0) / (x1 - x0) + y0

def read_overrides(override_files):
    """Read files in override directory."""
    filenames = glob_file_list(override_files)
    filenames.sort()
    dict_list = [load_json_from_file(filename) for filename in filenames]
    override_dict = merge_dicts(*dict_list)
    return override_dict

def make_namedtuple(**kwargs):
    """Quickly make a namedtuple instance.
    Alternatively, use pd.Series(dict())."""
    cls = namedtuple('namedtuple', kwargs.keys())
    return cls(*kwargs.values())

def instance_method_wrapper(obj, method_name, *args, **kwargs):
    """Wraps an object instance method within a static method.
    This can be used when pickling is needed such as for multiprocessing.
    Example:
        from sklearn.externals.joblib import Parallel, delayed
        pjobs = [delayed(instance_method_wrapper)(self, 'instance_method_name', arg1, arg2) for i in range(3)]
        return Parallel(n_jobs=1)(pjobs)
    """
    method = getattr(obj, method_name)
    return method(*args, **kwargs)

def enable_logging(level=logging.INFO, warnlevel=logging.WARN):
    """Enable logging to iPython notebook."""
    rootLogger = logging.getLogger()
    rootLogger.setLevel(level)
    while rootLogger.handlers:
        rootLogger.removeHandler(rootLogger.handlers[0])
    errHandler = logging.StreamHandler(sys.stderr)
    errHandler.setLevel(warnlevel)
    errHandler.setFormatter(logging.Formatter('[%(levelname)-5.5s] [%(module)s] %(message)s'))
    rootLogger.addHandler(errHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(level)
    consoleHandler.setFormatter(logging.Formatter('[%(levelname)-5.5s] [%(module)s] %(message)s'))
    rootLogger.addHandler(consoleHandler)

def disable_logging():
    enable_logging(level=logging.WARN)
    
#
# End
#
