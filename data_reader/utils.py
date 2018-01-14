import os
import codecs

def open_read(file_path, mode="r"):
    fread = codecs.open(file_path, mode=mode, encoding="utf8")
    return fread

