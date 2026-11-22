#!/usr/bin/env python3
"""Extract diagnostic archives with libarchive using only Python stdlib.

Regular files/directories only, no absolute/traversal paths or overwrites.
Preserves the original archive and records hashes for reproducible analysis.
"""
import argparse,ctypes,ctypes.util,hashlib,json,shutil,stat
from pathlib import Path,PurePosixPath

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
 source=args.archive.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 lib=ctypes.CDLL(ctypes.util.find_library('archive'));lib.archive_read_new.restype=ctypes.c_void_p
 for n in ['archive_read_support_filter_all','archive_read_support_format_all','archive_read_free']:getattr(lib,n).argtypes=[ctypes.c_void_p]
 lib.archive_read_open_filename.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_size_t]
 lib.archive_read_next_header.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p)]
 lib.archive_entry_pathname.argtypes=[ctypes.c_void_p];lib.archive_entry_pathname.restype=ctypes.c_char_p
 lib.archive_entry_size.argtypes=[ctypes.c_void_p];lib.archive_entry_size.restype=ctypes.c_longlong
 lib.archive_entry_filetype.argtypes=[ctypes.c_void_p];lib.archive_entry_filetype.restype=ctypes.c_int
 lib.archive_read_data.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];lib.archive_read_data.restype=ctypes.c_ssize_t
 handle=lib.archive_read_new();lib.archive_read_support_filter_all(handle);lib.archive_read_support_format_all(handle)
 entries=[];total=0;buffer=ctypes.create_string_buffer(1024*1024)
 try:
  if lib.archive_read_open_filename(handle,str(source).encode(),10240)!=0:raise ValueError('Cannot open archive')
  entry=ctypes.c_void_p()
  while True:
   rc=lib.archive_read_next_header(handle,ctypes.byref(entry))
   if rc==1:break
   if rc!=0:raise ValueError('Archive header error')
   name=lib.archive_entry_pathname(entry).decode();path=PurePosixPath(name)
   if path.is_absolute() or '..' in path.parts or '\\' in name or not path.parts:raise ValueError('Unsafe archive path')
   target=out.joinpath(*path.parts);kind=lib.archive_entry_filetype(entry)
   if kind==stat.S_IFDIR:target.mkdir(parents=True,exist_ok=True);continue
   if kind!=stat.S_IFREG:raise ValueError('Only regular files are accepted')
   size=lib.archive_entry_size(entry)
   if size<0 or size>500*1024**2 or total+size>1024**3:raise ValueError('Diagnostic archive size bound exceeded')
   target.parent.mkdir(parents=True,exist_ok=True);sha=hashlib.sha256();written=0
   with target.open('xb') as f:
    while True:
     n=lib.archive_read_data(handle,buffer,len(buffer))
     if n==0:break
     if n<0:raise ValueError('Archive data error')
     written+=n
     if written>size:raise ValueError('Archive entry larger than declared')
     block=buffer.raw[:n];f.write(block);sha.update(block)
   if written!=size:raise ValueError('Archive entry size mismatch')
   total+=written;entries.append({'path':str(target.relative_to(out)),'bytes':written,'sha256':sha.hexdigest()})
 finally:lib.archive_read_free(handle)
 shutil.copyfile(source,out/'source_original.7z')
 manifest={'archive':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'entries':entries,'total_bytes':total}
 (out/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(f'Extracted {len(entries)} files, {total} bytes; archive SHA256 {manifest["sha256"]}')
if __name__=='__main__':main()
