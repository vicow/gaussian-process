# coding=utf8

import argparse
import datetime
import re
import sys


# state variables for displaying progress
_progress_start = None
_progress_nb_chars = -1


def get_total_seconds(td):
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 1e6) / 1e6


def progress(count, total):
    """Shows the progress of some task.
    """
    global _progress_start, _progress_nb_chars
    if count > 0:
        sys.stdout.write('\b' * _progress_nb_chars)
    else:
        _progress_start = datetime.datetime.now()

    msg = 'Done!'.ljust(_progress_nb_chars) + '\n'

    if count != total:
        msg = ('%.2f%%' % (count * 100.0 / (total - 1),)).rjust(6)
        time_elapsed = get_total_seconds(datetime.datetime.now() -
                                         _progress_start)
        progress = count * 1.0 / total
        if progress > 0:
            remaining = time_elapsed / progress - time_elapsed
            msg += ' (%s remaining)' % (sec2str(remaining),)
        msg = msg.ljust(_progress_nb_chars)

    _progress_nb_chars = len(msg)
    sys.stdout.write(msg)

    sys.stdout.flush()


def sec2str(duration):
    """Converts a duration in seconds to a readable string.
    """
    duration *= 1.0
    units = ['s', 'm', 'h', 'd']
    divider = [60, 60, 24]

    while len(divider) and duration > divider[0]:
        duration /= divider[0]
        divider.pop(0)
        units.pop(0)

    return '%d%s' % (duration, units[0])


class YearAction(argparse.Action):
    '''Action to clean years given as argument.

    It allows to give years in short format, such as 7 for 2007, and
    to always have the full year (2007) as value for the argument..
    '''
    def __call__(self, parser, namespace, values, option_string=None):
        if values < 2000:
            values += 2000
        setattr(namespace, self.dest, values)


def clean_str(string):
    string = string.replace(u'ç', 'c')
    return re.sub('(-[a-zA-Z]+)', '',
                  re.sub('[^a-z\\-A-Z]', '',
                         re.sub('\(.+\)', '', string.split()[0]))).lower()


class Candidate:
    def __init__(self, first, last, canton, year):
        self.first = clean_str(first)
        self.last = clean_str(last)
        self.canton = canton
        self.year = year

    def __unicode__(self):
        return u'[%s] %s, %s (%d)' % (self.canton,
                                      self.last,
                                      self.first,
                                      self.year)

    def __str__(self):
        return unicode(self).encode('utf8')

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        is_eq = (isinstance(other, self.__class__)
                 and self.first == other.first
                 and self.last == other.last
                 and self.canton == other.canton)
        # disambiguate special cases
        special_cases = [('markus', 'baumann', 'ZH'),
                         ('martin', 'stalder', 'ZH'),
                         ('ueli', 'maurer', 'ZH')]
        if (self.first, self.last, self.canton) in special_cases:
            is_eq = is_eq and (self.year == other.year)

        return is_eq

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return (13 * hash(self.clean(self.first)) + 11 * hash(self.last)
                + 7 * hash(self.canton) + self.year)
