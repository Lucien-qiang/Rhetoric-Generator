'''
   Created by zhiqliu on 2018/12/18.
'''

import logging
import logging.handlers
import os
import os.path

global g_log_inst


class Logger(object):
    _inst = None
    _level_dict = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET,
    }

    @classmethod
    def start(cls, log_path, name=None, level=None):
        if cls._inst is not None:
            return cls._inst

        fpath = '/'.join(log_path.split('/')[0: -1])
        if False == os.path.exists(fpath):
            os.mkdir(fpath)
        fmt = '[%(levelname)s] %(asctime)s, pid=%(process)d, src=%(pathname)s:%(lineno)d, %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        cls._inst = logging.getLogger(name)
        cls._inst.setLevel(Logger._level_dict['DEBUG'])

        handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=500 * (1 << 20), backupCount=8)
        fmtter = logging.Formatter(fmt, datefmt)
        handler.setFormatter(fmtter)

        cls._inst.addHandler(handler)

    @classmethod
    def get(cls):
        return cls._inst


g_log_inst = Logger
