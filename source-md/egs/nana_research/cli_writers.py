from pathlib import Path
from typing import Dict

import kaldiio
import numpy

def file_writer_helper(wspecifier:str,filetype:str='mat',
                      compress:bool=False,
                      compression_method:int=2):
    """Write matrices in kaldi style

    Args:
      wspecifier:e.g:ark,scp:out.ark,out.scp
      filetype:"mat" is kaldi-matrix,
      compress:Compress or not
      compression_method:Specify compression level

    Write in kaldi-matrix-ark with "kaldi-scp"file:

    >>>with file_writer_helper('ark,t,scp:out.ark,out.scp') as f:
    >>> f['uttid'] = array

    This "scp" has the following format:
   
       uttidA out.ark:1234
       uttidB out.ark:2222
    where,1234 and 2222 points the starting byte address of the matrix.
    (For detail, see official documentation of kaldi)
    """

    if filetype == 'mat':
        return KaldiWriter(wspecifier,compress=compress,compression_method=compression_method)
    else:
        raise NotImplementedError(f'filetype={filetype}')

class BaseWriter:
    def __setitem__(self,key,value):
        raise NotImplementedError
     
    def __enter__(self):
        return self

    def __exit__(self,exc_type,exc_val,exc_tb):
        self.close()

    def close(self):

        try:
            self.writer.close()
        except Exception:
            pass

        if self.writer_scp is not None:
            try:
                self.writer_scp.close()
            except Exception:
                pass


class KaldiWriter(BaseWriter):
    def __init__(self,wspecifier,compress=False,compression_method=2):
        self.writer = kaldiio.WriteHelper(wspecifier)
        self.writer_scp = None
    def __setitem__(self, key,value):
        self.writer[key] = value

def parse_wspecifier(wspecifier:str) -> Dict[str, str]:
    """Parse wspecifier to dict

    Examples:
        >>>parse_wspecifier('ark,t,scp:out.ark,out.scp')
       {'ark,t':'out.ark','scp':'out.scp'}

    """
    ark_scp,filepath = wspecifier.split(':',1)
    if ark_scp not in ['ark','scp,ark','ark,scp','ark,t,scp']:
        raise ValueError(
          '{} is not allowed:{}'.format(ark_scp,wspecifier))
    ark_scps = ark_scp.split(',')
    filepaths = filepath.split(',')
    if len(ark_scps) != len(filepaths)+1:
        raise ValueError(
          'Mismatch:{} and {}'.format(ark_scp,filepath))
    spec_dict = dict(zip(ark_scps,filepaths))
    return spec_dict


