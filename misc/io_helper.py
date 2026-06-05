# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil

from misc.dist_helper import get_rank


class pathmgr:  # Make naming consistent
    @staticmethod
    def isfile(path):
        return os.path.isfile(path)

    @staticmethod
    def isdir(path):
        return os.path.isdir(path)

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    @staticmethod
    def ls(path):
        return os.listdir(path)

    @staticmethod
    def copy_from_local(src, tgt, overwrite=True):
        return shutil.copy2(src, tgt)

    @staticmethod
    def copy(src, tgt, overwrite=True):
        return shutil.copy2(src, tgt)

    @staticmethod
    def open(*args, **kwargs):
        return open(*args, **kwargs)

    @staticmethod
    def mkdirs(path):
        return os.makedirs(path, exist_ok=True)

    @staticmethod
    def get_local_path(path, force=True):
        return path


def may_download_to_local(path, subfolder=None):
    if subfolder is not None:
        path = os.path.join(path, subfolder)
    assert os.path.isdir(path) or os.path.isfile(path)
    return path


def mkdirs(dirpath, is_main_process_only=True):
    if is_main_process_only:
        if get_rank() == 0 and (not pathmgr.isdir(dirpath)):
            pathmgr.mkdirs(dirpath)
    else:
        if not pathmgr.isdir(dirpath):
            pathmgr.mkdirs(dirpath)
    return dirpath
