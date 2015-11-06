#!/usr/bin/env python

#  Copyright 2013, Haz-Edine Assemlal

#  This file is part of PYEZMINC.
# 
#  PYEZMINC is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, version 2.
# 
#  PYEZMINC is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with PYEZMINC.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division

import subprocess as sp
from operator import mul
import numpy as np
import os
import re
from datetime import datetime
from time import gmtime, strftime
import sys

import pyezminc


def gzip(fname=None, verbose=False):
    """ Gzip a file

    :param string fname: filename to gzip
    :returns: the filename of the gzipped file
    :rtype: string
    """
    sp.check_call(['gzip', '-f', fname])
    return fname + '.gz'


def mincinfo(header, field):
    """Gets data from a dictionary using a dotted accessor-string"""
    h = header
    for chunk in field.split(':'):
        h = h.get(chunk, {})
    if h == {}:
        raise Exception("field not found in header: '{0}'".format(field))
    return h


class Image(object):
    ''' Class to deal with minc files.

    :param data: image array
    :type data: numpy.nparray
    :param name: file name
    :type name: str
    :param history: brief history of modifications made to the image
    :type history: list of str
    :param header: header information
    :type header: dict
    '''

    def __init__(self, fname=None, data=None, autoload=True, dtype=np.float64, *args, **kwargs):
        self._dtype = dtype
        if pyezminc is None:
            raise ImportError("pyezminc not found")

        self.__wrapper = pyezminc.EZMincWrapper()
        self._history = ''
        self.data = data
        self.name = fname
        self._from_file = False
        self._header = None
        self._minc_dimensions_order = ('xspace', 'yspace', 'zspace', 'time')
        self._np_dimensions_order = ('time', 'zspace', 'yspace', 'xspace')

        if autoload and data is None and self.name:
            self.load(self.name, *args, **kwargs)

        if fname:
            self.history = self.__wrapper.history().rstrip('\n').split('\n')

    def load(self, name=None, with_nan=False, metadata_only=False, *args, **kwargs):
        '''Load a MINC image file, and put the content into self.data
        
        :param name: the filename of the MINC image. If none, then the value is self.name
        :type name: str
        :param with_nan: authorize NaN values
        :type with_nan: bool
        :param metadata_only: load the header only
        :type metadata_only: bool
        '''

        if name is None:
            name = self.name
        if not os.path.exists(name):
            raise IOError('file does not exist', name)

        self._from_file = True
        self.__wrapper.load(name, dtype=self._dtype, metadata_only=metadata_only)
        # catch NaN values
        if not metadata_only and np.any(self.data == -sys.float_info.max):
            if not with_nan:
                raise Exception("NaN value detected in '{0}'".format(name))
            else:
                self.data = np.where(self.data == -sys.float_info.max, np.nan, self.data).astype(self._dtype)

    def save(self, name=None, like=None, *args, **kwargs):
        '''Save the image to a MINC file.

        :param like: a reference filename for the MINC header.
        :type like: str
        '''
        if name is None:
            name = self.name
 
        if not like:
            if not self.name:
                raise Exception('like or name options have to be defined')
            like = self.name

        if not os.path.isfile(like):
            raise Exception("Cannot like from non existing file: '{0}'".format(like))

        compress = False
        if os.path.splitext(name)[1] == '.gz':
            name = os.path.splitext(name)[0]
            compress = True

        if self._dtype != self.data.dtype:
            raise Exception("Cannot save image because non consistent dtype, '{0}' != '{1}'".format(self._dtype, self.data.dtype))
            
        self.__wrapper.save(name, imitate=like, dtype=self._dtype, history=self.history)

        if compress:
            gzip(name)

    @property
    def data(self):
        ''' Return the data array '''
        return self.__wrapper.data

    @data.setter
    def data(self, value):
        ''' Map self.data to self.__wrapper.data '''
        self.__wrapper.data = value

    @property
    def header(self):
        if self._from_file and self._header is None:
            try:
                self._header = self.__wrapper.parse_header()
            except Exception as e:
                print("MINC header exception to be fixed: {0}".format(e))
        return self._header

    @header.setter
    def header(self, value):
        raise Exception('not yet implemented')
        
    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, value):
        self._history = value

    def append_history(self, brief_description):
        stamp = strftime("%a %b %d %T %Y>>>", gmtime())
        self.history.append(stamp+(' '.join(brief_description)))

    @property
    def dimensions_order(self):
        if self._from_file:
            idx = len(self._np_dimensions_order) - self.__wrapper.nb_dim()
            return self._np_dimensions_order[idx:]

    @property
    def voxel_spacing(self):
        if self._from_file:
            return tuple(
                self.__wrapper.nspacing(i+1)
                for i in reversed(range(self.__wrapper.nb_dim())))
        raise Exception('The voxel size is not defined')
    
    @property
    def dimensions_size(self):
        if self._from_file:
            dimensions_size = tuple(
                self.__wrapper.ndim(i+1)
                for i in reversed(range(self.__wrapper.nb_dim())))
        else:
            dimensions_size = self.data.shape
        return dimensions_size
    
    @property
    def world_start(self):
        if self._from_file:
            return tuple(
                self.__wrapper.nstart(i+1)
                for i in reversed(range(self.__wrapper.nb_dim())))
        raise Exception('The voxel coordinates are not defined')

    def volume(self, nb_voxels):
        '''Given a number of voxels, it returns the volume.'''
        if self._from_file:
            one_voxel_cc = [self.__wrapper.nspacing(i+1)
                            for i in range(self.__wrapper.nb_dim())]
            return nb_voxels * reduce(mul, one_voxel_cc)
        else:
            raise Exception('The voxel size is not defined, cannot calculate the volume')

    @property
    def world_direction_cosines(self):
        ''' Get MINC direction cosines. If not defined, then assume standard orthogonal base'''

        cosines = []
        for i in reversed(range(self.__wrapper.nb_dim())):
            if self.__wrapper.have_dir_cos(i+1):
                cosines.append(tuple(
                    self.__wrapper.ndir_cos(i+1, j)
                    for j in range(self.__wrapper.nb_dim())))
            else:
                cosines.append(tuple(
                    1 if j == i else 0
                    for j in range(self.__wrapper.nb_dim())))
        return tuple(cosines)
    
    def voxel_to_world(self, voxel):
        ''' Map voxel coordinates to world coordinates'''

        world_tmp = [voxel[i]*self.voxel_spacing[i] + self.world_start[i]
                     for i in range(self.__wrapper.nb_dim())]
        cosines_transpose = np.transpose(np.asarray(self.world_direction_cosines))
        world = []
        world = tuple(
            sum(p*q for p, q in zip(world_tmp, cosines_transpose[i]))
            for i in range(self.__wrapper.nb_dim()))
        return world

    def scanner_field_strength(self, field='dicom_0x0018:el_0x0087'):
        out = mincinfo(self.header, field)
        val = re.search('([0-9\.]*)'.format(field), out).group(1)
        # Make sure this is a number
        try:
            float(val)
        except:
            raise Exception('Could not extract the field strength as a float value: "{0}"'.format(val))
        return float(val)

    def acquisition_date(self):
        #::StartAcquiring[Acq-Time ]--->Acquiring--->::StartStoring[ContentTime]-->Storing.    
        #(0008,0013) TM [171809.637000]                          #  14, 1 InstanceCreationTime
        #(0008,0020) DA [20111006]                               #   8, 1 StudyDate
        #(0008,0021) DA [20111006]                               #   8, 1 SeriesDate
        #(0008,0022) DA [20111006]                               #   8, 1 AcquisitionDate
        #(0008,0023) DA [20111006]                               #   8, 1 ContentDate
        #(0008,0030) TM [165732.988000]                          #  14, 1 StudyTime
        #(0008,0031) TM [171609.633000]                          #  14, 1 SeriesTime
        #(0008,0032) TM [171736.510000]                          #  14, 1 AcquisitionTime
        #(0008,0033) TM [171809.637000]                          #  14, 1 ContentTime

        field_date = ['dicom_0x0008:el_0x0020',
                      'dicom_0x0008:el_0x0021',
                      'dicom_0x0008:el_0x0022',
                      'dicom_0x0008:el_0x0023',
                      'dicom_0x0008:el_0x0012']
        field_time = ['dicom_0x0008:el_0x0032',
                      'dicom_0x0008:el_0x0031',
                      'dicom_0x0008:el_0x0033']
        field_datetime = ['acquisition:start_time', 'study:start_time']

        # First try to get date and time together from standard MINC fields
        header = self.header
        for field in field_datetime:
            try:
                out = mincinfo(header, field)
                m = re.search('(?P<year>[0-9]{4})(?P<month>[0-9]{2})(?P<day>[0-9]{2})\s+(?P<hour>[0-9]{2})(?P<minute>[0-9]{2})(?P<second>[0-9]{2})\.?(?P<microsecond>[0-9]+)?', out).groupdict()
            except Exception:
                continue
            for k in m.keys():
                if m[k] is None:
                    del m[k]
                else:
                    m[k] = int(m[k])
            return datetime(**m)

        # If failed, first get date time from dicom fields
        date = None
        for field in field_date:
            try:
                out = mincinfo(header, field)
                m = re.search('(?P<year>[0-9]{4})(?P<month>[0-9]{2})(?P<day>[0-9]{2})', out).groupdict()
            except Exception:
                continue
            for k in m.keys():
                if m[k] is None:
                    del m[k]
                else:
                    m[k] = int(m[k])
            date = datetime(**m)
            break

        # Then get scan time
        for field in field_time:
            try:
                out = mincinfo(header, field)
                m = re.search('(?P<hour>[0-9]{2})(?P<minute>[0-9]{2})(?P<second>[0-9]{2})\.?(?P<microsecond>[0-9]+)?', out).groupdict()
            except Exception:
                continue
            for k in m.keys():
                if m[k] is None:
                    del m[k]
                else:
                    m[k] = int(m[k])
            date.replace(**m)
            break
        if date:
            return date
        raise Exception('Could not extract acquisition date from {0}'.format(self.name))

    def to_Label(self):
        """Convert a copy of this instance as a Label instance.
        :rtype: a minc.Label instance.
        """
        return Label(data=self.data.astype(np.int32))
                

class Label(Image):
    def __init__(self, fname=None, data=None, autoload=True, dtype=np.int32, *args, **kwargs):
        self.include_not_label = False
        super(Label, self).__init__(fname=fname, data=data, autoload=autoload, dtype=dtype, *args, **kwargs)

    def to_Mask(self):
        """Convert a copy of this instance as a Mask instance.
        :rtype: a minc.Mask instance.
        """
        return Mask(data=self.data, mnc2np=True)
        

class Mask(Image):
    def __init__(self, fname=None, data=None, autoload=True, dtype=np.int32, mnc2np=True, *args, **kwargs):
        self.mnc2np = mnc2np
        super(Mask, self).__init__(fname=fname, data=data, autoload=autoload, dtype=dtype, *args, **kwargs)
        if mnc2np and data is not None and not fname:
            self.data = self._minc_to_numpy(self.data)
    
    def _minc_to_numpy(self, data):
        return np.logical_not(data.astype(np.bool)).astype(np.dtype)
    
    def _numpy_to_minc(self, data):
        return np.logical_not(data.astype(np.bool)).astype(self._dtype)
        
    def _load(self, name, *args, **kwargs):
        '''Load a file'''
        super(Mask, self)._load(name, *args, **kwargs)
        if self.mnc2np:
            self.data = self._minc_to_numpy(self.data)
        
    def _save(self, name=None, like=None, *args, **kwargs):
        old_data = self.data
        try:
            self.data = self._numpy_to_minc(self.data)
            super(Mask, self)._save(name=name, like=like, *args, **kwargs)
        finally:
            self.data = old_data


def main(argv=None):
    DATA_PATH = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'test')

    label_name = os.path.join(DATA_PATH, 'trial_site00_subject_00_screening_gvf_ISPC-stx152lsq6.mnc.gz')
    l = minc.Label(label_name)
    
    img_name = os.path.join(DATA_PATH, 'trial_site01_subject_00_screening_t2w.mnc.gz')
    img = minc.Image(img_name)

    print('All appears OK')

if __name__ == '__main__':
    main()

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
