import os
import cv2
import math
import shutil
import dask as da
import numpy as np
import xarray as xr
import holoviews as hv
from typing import List
from tqdm.notebook import tqdm
from tifffile import TiffFile, imread
from scipy.signal import periodogram, find_peaks


def vid_frame(frame, varr_ref):
    """
    Used to create a dictionary of images for a scrollbar in HoloViews. 
    Enables detection of single stripes. This is an optional feature.
    """
    return hv.Image(varr_ref.sel(frame=frame))


def load_tif_lazy(fname):
    data = TiffFile(fname)
    f = len(data.pages)

    fmread = da.delayed(load_tif_perframe)
    flist = [fmread(fname, i) for i in range(f)]

    sample = flist[0].compute()
    arr = [
        da.array.from_delayed(fm, dtype=sample.dtype, shape=sample.shape)
        for fm in flist
    ]
    return da.array.stack(arr, axis=0)


def load_tif_perframe(fname, fid):
    return imread(fname, key=fid)


def load_avi_lazy(fname):
    cap = cv2.VideoCapture(fname)
    f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fmread = da.delayed(load_avi_perframe)
    flist = [fmread(fname, i) for i in range(f)]
    sample = flist[0].compute()
    arr = [
        da.array.from_delayed(fm, dtype=sample.dtype, shape=sample.shape)
        for fm in flist
    ]
    return da.array.stack(arr, axis=0)


def load_avi_perframe(fname, fid):
    cap = cv2.VideoCapture(fname)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, fm = cap.read()
    if ret:
        return np.flip(cv2.cvtColor(fm, cv2.COLOR_RGB2GRAY), axis=0)
    else:
        print(f'frame read failed for frame {fid}')
        return np.zeros((h, w))


def factors(x: int) -> List[int]:
    """
    Compute all factors of an interger.

    Parameters
    ----------
    x : int
        Input

    Returns
    -------
    factors : List[int]
        List of factors of `x`.
    """
    return [i for i in range(1, x + 1) if x % i == 0]


def group_consecutives(vals, step=1):
    """
    Return list of consecutive lists of numbers from vals (number list).
    """
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


def no_stripes_frames(varr, thresh):
    """
    Used to decrease run time of no_stripes_in_frame function.
    """
    return xr.apply_ufunc(
        no_stripes_in_frame,
        varr,
        input_core_dims=[['height', 'width']],
        vectorize=True,
        dask='parallelized',
        kwargs={'thresh': thresh},
        output_dtypes=[bool])


def no_stripes_in_frame(x, thresh=18): # thresh argument important for detection, 13-18 works well
    """ 
    Used to detect stripes in a frame. Creates a periodogram and, if that frame has enough detected peaks, will be considered a bad frame.
    Args:
        x : numpy.ndarray
            single frame from video
        thresh : int
            threshold used to determine peaks
            values between 13 and 18 work well.
    """
    y = x.mean(axis=1)
    f, Pxx_spec = periodogram(y, 1)
    peaks = find_peaks(np.sqrt(Pxx_spec), height=thresh)[0]
    return not any(f[peaks] > 0.03)


def get_filename(dpath, bad_frames, framesPerFile=1000):
    """ 
    Finds the file names associated with each bad frame.
    Args:
        dpath : str
            path to .avi files
        bad_frames : numpy.ndarray
            an array of all the frames that are bad in the recording
        framesPerFile : int
            by default 1000. This is set in the Miniscope config file and shouldn't be touched.
    Returns:
        fnames : list
            list of file names with bad frames
        relative_frame_numbers : list of lists
            the first list is the same length as the number of files with bad frames
            the second list is all the frames that are bad in that file, relative from 0 to 1000
    """
    # Get the file names associated with each frame. 
    vid_numbers = np.unique([math.floor(f/framesPerFile) for f in bad_frames])
    fnames = [os.path.join(dpath, str(n) + '.avi') for n in vid_numbers]
    # Get the frame number within that video file.
    relative_frame_numbers = []
    for n in vid_numbers:
        if n == 0:
            relative_frame_numbers.append([f for f in bad_frames[bad_frames < 1000]])
        else:
            quotient, remainder = np.divmod(bad_frames, n*framesPerFile)
            relative_frame_numbers.append(remainder[(quotient==1) & (remainder < framesPerFile)])
    return fnames, relative_frame_numbers


def pad_frame(frame, video_cropped):
    """ 
    Adds zeros to missing part of the matrix if the recording has been cropped.
    Args:
        frame : numpy.ndarray
            single frame from the recording
        video_cropped : dict
            dictionary that specifies height and width values
            will not be None if the recording was cropped
    Returns:
        fullframe : numpy.ndarray
            a 608 x 608 array, which is the default Miniscope frame size
    """
    fullframe = np.zeros((608, 608), dtype='uint8')
    fullframe[video_cropped['h'][0]:video_cropped['h'][1], video_cropped['w'][0]:video_cropped['w'][1]] = frame
    return fullframe


def correct_stripes(frame, video_cropped, offset, buffsize=8184, buffertofix=1):
    """ 
    Fix mis-aligned data by rolling the array by the amount provided through offset.
    
    Args:
        frame : numpy.ndarray
            single frame from the recording
        video_cropped : dict
            dictionary that specifies height and width values
            will not be None if the recording was cropped
        offset : int
            amount to roll the flattened array by. Should be the buffer size 8184
        buffersize : int
            by default 8184
        buffertofix : int
            should be either 1 or 2. I've only ever seen the first (1) buffer having an issue.
    Returns:
        frame_fixed : numpy.ndarray
            shifted data back in the correct frame shape
    """
    # Pad croppped frames to equal original frame size
    if video_cropped is not None:
        frame = pad_frame(frame, video_cropped)
    # Correct buffer
    flatframe = frame.flatten()
    buffermask = (np.arange(frame.shape[0] * frame.shape[1]) // buffsize % 2) + 1 # labels first and second buffer
    originalbuffer = flatframe[buffermask==buffertofix] # extract buffer to fix
    fixedbuffer = np.roll(originalbuffer, offset)
    flatframe[buffermask==buffertofix] = fixedbuffer
    frame_fixed = flatframe.reshape((608, 608))
    # Remove padding
    if video_cropped is not None:
        frame_fixed = frame_fixed[video_cropped['h'][0]:video_cropped['h'][1], video_cropped['w'][0]:video_cropped['w'][1]]
    return frame_fixed


def fix_video(fnames, frame_numbers, video_cropped, offset, folder_name='originals', buffsize=8184, buffertofix=1, compressionCodec='FFV1'):
    """ 
    This function will take the detected bad frames, re-align the data, and save the corrected video without overwriting the originals.
    Args:
        fnames : list
            list of file names that contain bad frames
        frame_numbers : list of lists
            the first list is the same length as the number of files with bad frames
            the second list is all the frames that are bad in that file, relative from 0 to 1000
        video_cropped : dict
            dictionary that specifies height and width values
            will not be None if the recording was cropped
        offset : int
            amount to roll the flattened array by. Should be the buffer size 8184
        folder_name : str
            name of folder to move original files to to prevent overwriting data
        buffersize : int
            by default 8184
        buffertofix : int
            should be either 1 or 2. I've only ever seen the first (1) buffer having an issue.
        compressionCodec : str
            what compression mode to use. Depends on what compression was used to collect the data in the Miniscope software.
            by default FFV1
    Raise:
        FileExistsError
            if there is already a folder with the same name, the function will abort to prevent overwriting data
    """
    folder = os.path.join(os.path.split(fnames[0])[0], folder_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
        print(f'Created {folder}')
    for idx, video in enumerate(fnames):
        cap = cv2.VideoCapture(video)
        rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        video_name = os.path.split(video)[1]
        move_video_to = os.path.join(folder, video_name)
        if not os.path.exists(move_video_to):
            shutil.move(video, move_video_to)
            print(f'Moved {video} to {move_video_to}')
        else:
            raise FileExistsError('The folder is already storing an original file. Aborting to prevent overwrite.')
        print ('Beginning video re-write...')
        cap = cv2.VideoCapture(move_video_to)
        writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*compressionCodec), 30, (int(cols), int(rows)), isColor=False)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
        for frame_number in tqdm(np.arange(total_frames)):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame_number in frame_numbers[idx]:
                    frame = correct_stripes(frame, video_cropped, offset=offset, buffsize=buffsize, buffertofix=buffertofix)
                writer.write(frame)
        writer.release()
        cap.release()


def rewrite_video(fnames, frame_numbers, folder_name='failed_to_fix', compressionCodec='FFV1'):
    """ 
    Replaces unfixable frames with a good frame the frame before a bad frame.
    Args:
        fnames : list
            list of file names that contain bad frames
        frame_numbers : list of lists
            the first list is the same length as the number of files with bad frames
            the second list is all the frames that are bad in that file, relative from 0 to 1000
        folder_name : str
            name of folder to move original files to to prevent overwriting data
        compressionCodec : str
            what compression mode to use. Depends on what compression was used to collect the data in the Miniscope software.
            by default FFV1
    Raise:
        FileExistsError
            if there is already a folder with the same name, the function will abort to prevent overwriting data
    """
    folder = os.path.join(os.path.split(fnames[0])[0], folder_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
        print(f'Created {folder}')
    codec = cv2.VideoWriter_fourcc(*compressionCodec)
    # For each video...
    for video, frames in zip(fnames, frame_numbers):
        print(f'Rewriting {video}')
        cap = cv2.VideoCapture(video)
        rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        chunk_number = 0
        fname = os.path.split(video)[1]
        move_fpath = os.path.join(folder, fname)
        # Move the original file. 
        if not os.path.exists(move_fpath):
            shutil.move(video, move_fpath)
            print(f'Moved {video} to {move_fpath}')
        else:
            raise FileExistsError('The folder is already storing an original file. Aborting to prevent overwrite.')
        cap = cv2.VideoCapture(move_fpath)
        writeFile = cv2.VideoWriter(video, codec, 60, (cols, rows), isColor=False)
        # Group the frame numbers into chunks. Get the frame number that will replace all the bad frames.
        # This is the frame right before the first bad frame in each chunk.
        frame_chunks = group_consecutives(frames)
        frame_chunks = [np.arange(chunk[0], chunk[-1] + 2) for chunk in frame_chunks]
        replacement_frame_number = [frames[0] - 1 for frames in frame_chunks]
        # For each frame in the video...
        for frame_number in tqdm(np.arange(total_frames)):
            ret, frame = cap.read()
            frame = frame[:, :, 1]
            if ret:
                # If it's a replacement frame, store it. 
                if frame_number in replacement_frame_number:
                    replacement_frame = frame
                    chunk_number = replacement_frame_number.index(frame_number)
                # If it's a bad frame, replace it with the replacement frame. 
                elif frame_number in frame_chunks[chunk_number]:
                    frame = replacement_frame
                writeFile.write(np.uint8(frame))
            else:
                break
        writeFile.release()
        cap.release()
    cv2.destroyAllWindows()