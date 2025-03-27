#!/usr/bin/env python
"""
Python code for extracting frames from video file and synthesizing fake DVS
events from this video after SuperSloMo has generated interpolated
frames from the original video frames.

@author: Tobi Delbruck, Yuhuang Hu, Zhe He
@contact: tobi@ini.uzh.ch, yuhuang.hu@ini.uzh.ch, zhehe@student.ethz.ch
"""
# todo refractory period for pixel

import glob
import argparse
import sys
import time
import argcomplete
import cv2
import numpy as np
import os
from tempfile import TemporaryDirectory
import json
import torch
from v2ecore.v2e_utils import save_array_to_h5
from v2ecore.v2e_utils import read_image, \
    check_lowpass, v2e_quit, group_videos_by_name
from v2ecore.v2e_utils import set_output_dimension
from v2ecore.v2e_utils import set_output_folder
from v2ecore.v2e_utils import ImageFolderReader
from v2ecore.v2e_args import v2e_args, write_args_info, SmartFormatter
from v2ecore.v2e_args import v2e_check_dvs_exposure_args
from v2ecore.v2e_args import NO_SLOWDOWN
from v2ecore.renderer import EventRenderer, ExposureMode
from v2ecore.slomo import SuperSloMo
from v2ecore.emulator import EventEmulator
from v2ecore.v2e_utils import mat_to_mp4
import sys
import h5py
import tqdm 
from multiprocessing import Pool
## get current directory
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{path}/v2ecore/')

import logging
import time
from typing import Optional, Any

logging.basicConfig(level=logging.ERROR, filename='v2e.log', filemode='w')
root = logging.getLogger()
LOGGING_LEVEL=logging.INFO
root.setLevel(LOGGING_LEVEL)  # todo move to info for production
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
logging.addLevelName(
    logging.DEBUG, "\033[1;36m%s\033[1;0m" % logging.getLevelName(
        logging.DEBUG)) # cyan foreground
logging.addLevelName(
    logging.INFO, "\033[1;34m%s\033[1;0m" % logging.getLevelName(
        logging.INFO)) # blue foreground
logging.addLevelName(
    logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(
        logging.WARNING)) # red foreground
logging.addLevelName(
    logging.ERROR, "\033[38;5;9m%s\033[1;0m" % logging.getLevelName(
        logging.ERROR)) # red background
logger = logging.getLogger(__name__)

# torch device
torch_device:str = torch.device('cuda' if torch.cuda.is_available() else 'cpu').type
logger.info(f'torch device is {torch_device}')
if torch_device=='cpu':
    logger.warning('CUDA GPU acceleration of pytorch operations is not available; '
                   'see https://pytorch.org/get-started/locally/ '
                   'to generate the correct conda install command to enable GPU-accelerated CUDA.')
print(f'torch device is {torch_device}')
# may only apply to windows
def test_file_path(file_path: str, processed_files) -> str:
    output_folder = file_path.split(".mp4")[0]
    output_folder = output_folder.replace("surreal", "processed_surreal")
    if "_depth" in file_path:
        ## remove the _depth from the file name
        output_folder = file_path.split("_depth")[0]
    output_file = output_folder + "/dvs.h5"
    # print(processed_files)
    # print()
    # print(output_file)
    # exit()
    return output_file not in processed_files and "_depth" not in file_path
def get_args():
    """ proceses input arguments
    :returns: (args_namespace,other_args,command_line) """
    parser = argparse.ArgumentParser(
        description='v2e: generate simulated DVS events from video.',
        epilog='Run with no --input to open file dialog', allow_abbrev=True,
        formatter_class=SmartFormatter)

    parser = v2e_args(parser)

    #  parser.add_argument(
    #      "--rotate180", type=bool, default=False,
    #      help="rotate all output 180 deg.")
    # https://kislyuk.github.io/argcomplete/#global-completion
    # Shellcode (only necessary if global completion is not activated -
    # see Global completion below), to be put in e.g. .bashrc:
    # eval "$(register-python-argcomplete v2e.py)"
    argcomplete.autocomplete(parser)

    (args_namespace,other_args) = parser.parse_known_args() # change to known arguments so that synthetic input module can take arguments
    command_line=''
    for a in sys.argv:
        command_line=command_line+' '+a
    return (args_namespace,other_args,command_line)
def set_args():
    (args,other_args,command_line) = get_args()
    args.timestamp_resolution=.003

    
    args.pos_thres=.15
    args.neg_thres=.15
    args.sigma_thres=0.03
    args.output_width = 346
    args.output_height = 260
    args.cutoff_hz=15
    args.batch_size=16
    # DVS exposure
    exposure_mode, exposure_val, area_dimension = \
        v2e_check_dvs_exposure_args(args)
    if exposure_mode == ExposureMode.DURATION:
        dvsFps = 1. / exposure_val
    # if exposure_mode == ExposureMode.DURATION:
    #     dvsNumFrames = np.math.floor(
    #         dvsFps*srcDurationToBeProcessed/args.input_slowmotion_factor)
    return args, other_args, command_line, exposure_mode, exposure_val, area_dimension, dvsFps
def set_vars(args, other_args, command_line):
    # set input fil

    # Set output width and height based on the arguments

    # Writing the info file
    infofile = write_args_info(args, args.output_folder, other_args,command_line)

    fh = logging.FileHandler(infofile,mode='a')
    fh.setLevel(LOGGING_LEVEL)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # define video parameters
    # the input start and stop time, may be round to actual
    # frame timestamp§§
    def is_float(element: Any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    if not args.start_time is None and not args.stop_time is None and is_float(args.start_time) and is_float(args.stop_time) and args.stop_time<=args.start_time:
        logger.error(f'stop time {args.stop_time} must be later than start time {args.start_time}')
        v2e_quit(1)    

    if not args.disable_slomo and args.auto_timestamp_resolution is False \
            and args.timestamp_resolution is None:
        logger.error(
            'if --auto_timestamp_resolution=False, '
            'then --timestamp_resolution must be set to '
            'some desired DVS event timestamp resolution in seconds, '
            'e.g. 0.01')
        v2e_quit()

    if args.auto_timestamp_resolution is True \
            and args.timestamp_resolution is not None:
        logger.info(
            f'auto_timestamp_resolution=True and '
            f'timestamp_resolution={args.timestamp_resolution}: '
            f'Limiting automatic upsampling to maximum timestamp interval.')

    if args.leak_rate_hz > 0 and args.sigma_thres == 0:
        logger.warning(
            'leak_rate_hz>0 but sigma_thres==0, '
            'so all leak events will be synchronous')
    
def tmp_npy(vid, emulator, source_frames_dir, vid_idx, max_frames ):
    
    num_frames = 0
    inputHeight = None
    inputWidth = None
    inputChannels = None
    if vid.start_frame > 0:
        logger.info('skipping to frame {}'.format(vid.start_frame))
        for i in range(vid.start_frame):
            if isinstance(vid.cap,ImageFolderReader):
                if i<vid.start_frame-1:
                    ret,_=vid.cap.read(skip=True)
                else:
                    ret, _ = vid.cap.read()
            else:
                ret, _ = vid.cap.read()
            if not ret:
                raise ValueError(
                    'something wrong, got to end of file before '
                    'reaching start_frame')

    logger.info(
        'processing frames {} to {} from video input'.format(
            vid.start_frame, vid.stop_frame))

    c_l=0
    c_r=None
    c_t=0
    c_b=None
    if vid.args.crop is not None:
        c=vid.args.crop
        if len(c)!=4:
            logger.error(f'--crop must have 4 elements (you specified --crop={vid.args.crop}')
            v2e_quit(1)

        c_l=c[0] if c[0] > 0 else 0
        c_r=-c[1] if c[1]>0 else None
        c_t=c[2] if c[2]>0 else 0
        c_b=-c[3] if c[3]>0 else None
        logger.info(f'cropping video by (left,right,top,bottom)=({c_l},{c_r},{c_t},{c_b})')
        
    
    if os.path.isdir(vid.file_path):  # folder input
        inputWidth = vid.cap.frame_width
        inputHeight = vid.cap.frame_height
        inputChannels = vid.cap.frame_channels
    else:
        inputWidth = int(vid.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        inputHeight = int(vid.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        inputChannels = 1 if int(vid.cap.get(cv2.CAP_PROP_MONOCHROME)) \
            else 3
    logger.info(
        'Input video {} has W={} x H={} frames each with {} channels'
        .format(vid.file_path, inputWidth, inputHeight, inputChannels))

    if (vid.output_width is None) and (vid.output_height is None):
        output_width = inputWidth
        output_height = inputHeight
        logger.warning(
            'output size ({}x{}) was set automatically to '
            'input video size\n    Are you sure you want this? '
            'It might be slow.\n Consider using\n '
            '    --output_width=346 --output_height=260\n '
            'to match Davis346.'
            .format(output_width, output_height))

        # set emulator output width and height for the last time
        emulator.output_width = output_width
        emulator.output_height = output_height

    video_frames = []
    for inputFrameIndex in range(max_frames):
        if inputFrameIndex < vid.srcNumFramesToBeProccessed: 
            # read frame
            ret, inputVideoFrame = vid.cap.read()

            num_frames+=1
            if ret==False:
                logger.warning(f'could not read frame {inputFrameIndex} from {vid.cap}')
                continue
            if inputVideoFrame is None or np.shape(inputVideoFrame) == ():
                logger.warning(f'empty video frame number {inputFrameIndex} in {vid.cap}')
                continue
            if not ret or inputFrameIndex + vid.start_frame > vid.stop_frame:
                break

            if vid.args.crop is not None:
                # crop the frame, indices are y,x, UL is 0,0
                if c_l+(c_r if c_r is not None else 0)>=inputWidth:
                    logger.error(f'left {c_l}+ right crop {c_r} is larger than image width {inputWidth}')
                    v2e_quit(1)
                if c_t+(c_b if c_b is not None else 0)>=inputHeight:
                    logger.error(f'top {c_t}+ bottom crop {c_b} is larger than image height {inputHeight}')
                    v2e_quit(1)

                inputVideoFrame= inputVideoFrame[c_t:c_b, c_l:c_r] # https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python

            if vid.output_height and vid.output_width and \
                    (inputHeight != vid.output_height or
                        inputWidth != vid.output_width):
                dim = (vid.output_width, vid.output_height)
                (fx, fy) = (float(vid.output_width) / inputWidth,
                            float(vid.output_height) / inputHeight)
                inputVideoFrame = cv2.resize(
                    src=inputVideoFrame, dsize=dim, fx=fx, fy=fy,
                    interpolation=cv2.INTER_AREA)
            if inputChannels == 3:  # color
                if inputFrameIndex == 0:  # print info once
                    logger.info(
                        '\nConverting input frames from RGB color to luma')
                # TODO would break resize if input is gray frames
                # convert RGB frame into luminance.
                inputVideoFrame = cv2.cvtColor(
                    inputVideoFrame, cv2.COLOR_BGR2GRAY)  # much faster

        # save frame into numpy records
        video_frames.append(inputVideoFrame)
        
    video_frames = np.stack(video_frames, axis=0)
    
    save_path = os.path.join( source_frames_dir,f"video_{str(vid_idx).zfill(4)}.h5")
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('video', data=video_frames)
    ## reload the file
    
    vid.cap.release()

def get_models(args, vids, exposure_mode, exposure_val, area_dimension, torch_device):
    srcFps = vids[0].srcFps
    srcFrameIntervalS = (1./srcFps)/args.input_slowmotion_factor
    slowdown_factor = NO_SLOWDOWN  # start with factor 1 for upsampling
    if args.disable_slomo:
        logger.warning(
            'slomo interpolation disabled by command line option; '
            'output DVS timestamps will have source frame interval '
            'resolution')
        # time stamp resolution equals to source frame interval
        slomoTimestampResolutionS = srcFrameIntervalS
    elif not args.auto_timestamp_resolution:
        
        slowdown_factor = int(
            np.ceil(srcFrameIntervalS/args.timestamp_resolution))
        slomoTimestampResolutionS = srcFrameIntervalS/slowdown_factor
        check_lowpass(args.cutoff_hz, 1/slomoTimestampResolutionS, logger)
    else:  # auto_timestamp_resolution
        if args.timestamp_resolution is not None:
            slowdown_factor = int(
                np.ceil(srcFrameIntervalS/args.timestamp_resolution))


    # the SloMo model, set no SloMo model if no slowdown
    if not args.disable_slomo and \
            (args.auto_timestamp_resolution or slowdown_factor != NO_SLOWDOWN):
        
        slomo = SuperSloMo(
            model=args.slomo_model,
            auto_upsample=args.auto_timestamp_resolution,
            upsampling_factor=slowdown_factor,
            videos= vids,
            vid_orig=None if args.skip_video_output else args.vid_orig,
            preview= not args.no_preview, batch_size=args.batch_size)



    if "depth" not in vids[0].file_path:

        emulator = EventEmulator(
            vids=vids,
            pos_thres=args.pos_thres, neg_thres=args.neg_thres,
            sigma_thres=args.sigma_thres, cutoff_hz=args.cutoff_hz,
            leak_rate_hz=args.leak_rate_hz, shot_noise_rate_hz=args.shot_noise_rate_hz, photoreceptor_noise=args.photoreceptor_noise,
            leak_jitter_fraction=args.leak_jitter_fraction,
            noise_rate_cov_decades=args.noise_rate_cov_decades,
            refractory_period_s=args.refractory_period,
            seed=args.dvs_emulator_seed,
            output_folder=args.output_folder, dvs_h5=args.dvs_h5, dvs_aedat2=args.dvs_aedat2, dvs_aedat4 = args.dvs_aedat4,
            dvs_text=args.dvs_text, show_dvs_model_state=args.show_dvs_model_state,
            save_dvs_model_state=args.save_dvs_model_state,
            output_width=args.output_width, output_height=args.output_height,
            device=torch_device,
            cs_lambda_pixels=args.cs_lambda_pixels, cs_tau_p_ms=args.cs_tau_p_ms,
            hdr=args.hdr,
            scidvs=args.scidvs,
            record_single_pixel_states=args.record_single_pixel_states,
            label_signal_noise=args.label_signal_noise
        )
        
        if args.dvs_params is not None:
            logger.warning(
                f'--dvs_param={args.dvs_params} option overrides your '
                f'selected options for threshold, threshold-mismatch, '
                f'leak and shot noise rates')
            emulator.set_dvs_params(args.dvs_params)

        # eventRenderer = EventRenderer(
        #     vids=vids,
        #     dvs_vid=args.dvs_vid, preview=not args.no_preview, full_scale_count=args.dvs_vid_full_scale,
        #     exposure_mode=exposure_mode,
        #     exposure_value=exposure_val,
        #     area_dimension=area_dimension,
        #     avi_frame_rate=args.avi_frame_rate)
        eventRenderer = None
    else:
        emulator, eventRenderer = None, None
    return emulator, eventRenderer, slomo, srcFrameIntervalS, slowdown_factor
def slowmo_upsampling(args, slomo, source_frames_dir, srcFrameIntervalS, slowdown_factor, logger):
    if slomo is not None and (args.auto_timestamp_resolution or slowdown_factor != NO_SLOWDOWN):
        # interpolated frames are stored to tmpfolder as
        # 1.png, 2.png, etc
        slow_mo_vids, interpTimes, avgUpsamplingFactor = slomo.interpolate_batch(source_frames_dir)
        avgTs = srcFrameIntervalS / avgUpsamplingFactor
        # check for undersampling wrt the
        # photoreceptor lowpass filtering

        if args.cutoff_hz > 0:
            check_lowpass(args.cutoff_hz, 1/avgTs, logger)

        # read back to memory
        # number of frames¨

    return slow_mo_vids, interpTimes
def event_sampling(vids_slowmo, vids, emulator, eventRenderer, interpTimes, args):
    import time
    start = time.time()
    nFrames = vids_slowmo.shape[1]
    # interpTimes is in units of 1 per input frame,
    # normalize it to src video time range
    real_dur = max([vid.srcVideoRealProcessedDuration for vid in vids])
    f = real_dur /(
        np.max(interpTimes)-np.min(interpTimes))
    # compute actual times from video times
    interpTimes = f*interpTimes
    # array to batch events for rendering to DVS frames
    events = np.zeros((0, 5), dtype=np.float32)
    # parepare extra steps for data storage
    # right before event emulation
    if args.ddd_output:
        emulator.prepare_storage(nFrames, interpTimes)

    # generate events from frames and accumulate events to DVS frames for output DVS video
    with torch.no_grad():
        for i in range(nFrames):
            fr = vids_slowmo[:, i]

            newEvents = emulator.generate_events(
                fr, interpTimes[i])
            
            if newEvents is not None and \
                    newEvents.shape[0] > 0 \
                    and not args.skip_video_output:
                events = np.append(events, newEvents, axis=0)
                events = np.array(events)
        
                # if i % args.batch_size == 0:
                #     eventRenderer.render_events_to_frames(
                #         events, height=args.output_height,
                #         width=args.output_width,
                #         batch_size = len(vids))
                #     events = np.zeros((0, 5), dtype=np.float32)
        # process leftover events
        to_save = events
        save_args = []
        for idx, vid in enumerate(emulator.vids):
            path = vid.dvs_npy  # a simple string path
            array = events[to_save[:,-1]==idx]
            save_args.append((path, array))

        with Pool(processes=16) as pool:
            pool.map(save_array_to_h5, save_args)
        
        # if len(events) > 0 and not args.skip_video_output:
        #     eventRenderer.render_events_to_frames(
        #         events, height=args.output_height, width=args.output_width, batch_size = len(vids))
    
class VideoInfos:
    def __init__(self, file_path, args):
        self.file_path = file_path
        self.output_folder = file_path.split(".mp4")[0]
        self.dvs_aedat4 = None
        if "_depth" in file_path:
            ## remove the _depth from the file name
            self.output_folder = file_path.split("_depth")[0]
        self.processed_folder = self.output_folder.replace("surreal", "processed_surreal")
        if "_depth" not in file_path:
            
            self.dvs_npy =  self.processed_folder + '/dvs.h5'
            self.video_dvs = self.processed_folder + '/dvs.mp4'
            self.dvs_times = self.processed_folder + '/dvs_frames_times.txt'
        self.video_output_file = None
        self.frame_times_output_file = None
        self.output_folder = self.output_folder.replace("surreal", "processed_surreal")
        self.args = args    
        self.vid_slomo = self.processed_folder + "/vid_slomo_depth.h5" if "_depth" in file_path else self.processed_folder + "/vid_slomo.h5"
        
        self.output_width, self.output_height = set_output_dimension(
        args.output_width, args.output_height,
        args.dvs128, args.dvs240, args.dvs346,
        args.dvs640, args.dvs1024,
        logger)

        set_output_folder(
        self.output_folder,
        self.file_path,
        args.unique_output_folder if not args.overwrite else False,
        args.overwrite,
        args.output_in_place,
        logger)
        self.extract_video_info()
    def extract_video_info(self):
        srcNumFramesToBeProccessed = 0
        input_file = self.file_path
        args = self.args
        if not os.path.isfile(input_file) and not os.path.isdir(input_file):
            logger.error('input file {} does not exist'.format(input_file))
            v2e_quit(1)
        if os.path.isdir(input_file):
            if len(os.listdir(input_file))==0:
                logger.error(f'input folder {input_file} is empty')
                v2e_quit(1)


        logger.info("opening video input file " + input_file)

        if os.path.isdir(input_file):
            if args.input_frame_rate is None:
                logger.error(
                    "When the video is presented as a folder, "
                    "The user must set --input_frame_rate manually")
                v2e_quit(1)

            cap = ImageFolderReader(input_file, args.input_frame_rate)
            srcFps = cap.frame_rate
            srcNumFrames = cap.num_frames

        else:
            cap = cv2.VideoCapture(input_file)
            srcFps = cap.get(cv2.CAP_PROP_FPS)
            srcNumFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if args.input_frame_rate is not None:
                logger.info(f'Input video frame rate {srcFps}Hz is overridden by command line argument --input_frame_rate={args.input_frame_rate}')
                srcFps=args.input_frame_rate

        if cap is not None:
            # set the output width and height from first image in folder, but only if they were not already set
            set_size = False
            if args.output_height is None and hasattr(cap,'frame_height'):
                set_size = True
                args.output_height = cap.frame_height
            if args.output_width is None and hasattr(cap,'frame_width'):
                set_size = True
                args.output_width = cap.frame_width
            if set_size:
                logger.warning(
                    f'From input frame automatically set DVS output_width={args.output_width} and/or output_height={args.output_height}. '
                    f'This may not be desired behavior. \nCheck DVS camera sizes arguments.')
                time.sleep(5);
            elif args.output_height is None or args.output_width is None:
                logger.warning(
                    'Could not read video frame size from video input and so could not automatically set DVS output size. \nCheck DVS camera sizes arguments.')

        # Check frame rate and number of frames
        if srcFps == 0:
            logger.error(
                'source {} fps is 0; v2e needs to have a timescale '
                'for input video'.format(input_file))
            v2e_quit()

        if srcNumFrames < 2:
            logger.warning(
                'num frames is less than 2, probably cannot be determined '
                'from cv2.CAP_PROP_FRAME_COUNT')

        srcTotalDuration = (srcNumFrames-1)/srcFps
        # the index of the frames, from 0 to srcNumFrames-1
        start_frame = int(srcNumFrames*(args.start_time/srcTotalDuration)) \
            if args.start_time else 0
        stop_frame = int(srcNumFrames*(args.stop_time/srcTotalDuration)) \
            if args.stop_time else srcNumFrames-1
        srcNumFramesToBeProccessed = stop_frame-start_frame+1
        # the duration to be processed, should subtract 1 frame when
        # calculating duration
        srcDurationToBeProcessed = (srcNumFramesToBeProccessed-1)/srcFps

        # redefining start and end time using the time calculated
        # from the frames, the minimum resolution there is
        start_time = start_frame/srcFps
        stop_time = stop_frame/srcFps
        srcVideoRealProcessedDuration = (stop_time-start_time) / \
                    args.input_slowmotion_factor
        self.start_time = start_time
        self.stop_time = stop_time
        self.cap = cap
        self.srcFps = srcFps
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.srcNumFramesToBeProccessed = srcNumFramesToBeProccessed
        self.srcDurationToBeProcessed = srcDurationToBeProcessed  
        self.srcVideoRealProcessedDuration = srcVideoRealProcessedDuration
        srcFrameIntervalS = (1./srcFps)/args.input_slowmotion_factor
        slowdown_factor = int(
            np.ceil(srcFrameIntervalS/args.timestamp_resolution))
        dict_time = {
        "srcFps":srcFps,
        "total_time":stop_time,
        "stop_frame":stop_frame,
        "upsampling_factor":slowdown_factor
        }
        with open(os.path.join(self.output_folder, "time.json"), 'w') as f:
            json.dump(dict_time, f)
    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        if self.video_output_file is not None and type(self.video_output_file) != str:
            self.video_output_file.release()
        if self.frame_times_output_file is not None and type(self.frame_times_output_file) != str:
            self.frame_times_output_file.close()

def main(file_paths: str):
    import time
    start = time.time()
    args, _, _, exposure_mode, exposure_val, area_dimension, dvsFps  = set_args()

    vids = [VideoInfos(file_path, args) for file_path in file_paths]
    
    
    emulator, eventRenderer, slomo, srcFrameIntervalS, slowdown_factor = get_models(args, vids, exposure_mode, exposure_val, area_dimension, torch_device)
    tmp_path = "/home/martin.barry/projects/tmp/"
    with TemporaryDirectory() as source_frames_dir:
        vid_idx = 0
        max_frames = max([vid.srcNumFramesToBeProccessed for vid in vids])
        for vid in vids:
            _ = tmp_npy(vid, emulator, source_frames_dir, vid_idx,max_frames)
                                            
            vid_idx += 1
            
            interpTimes = None
            # make input to slomo
        slow_mo_vids, interpTimes = slowmo_upsampling(args, slomo, source_frames_dir, 
                                                                srcFrameIntervalS, slowdown_factor, logger)
        if emulator is not None:   
            # compute times of output integrated frames
            event_sampling(slow_mo_vids, vids, emulator, eventRenderer, interpTimes, args)
            # eventRenderer.cleanup()
            emulator.cleanup()
       
    # Clean up
    for vid in vids:
        vid.cleanup()



    # sys.exit(0)

if __name__ == "__main__":
    import glob
    import os

    # data_path = "/home/martin.barry/projects/surreal/" ## change to your data path
    data_path = "/home/martin-barry/Downloads/surreal/"
    processed_files = glob.glob(os.path.join(data_path.replace("surreal", "processed_surreal"), "**/*.h5"), recursive = True)
    files = glob.glob(os.path.join(data_path, "**/*.mp4"), recursive = True)
    files = [file for file in files if test_file_path(file, processed_files)]
    ## shuffling the files
    # np.random.shuffle(files)

    batch_size = 2
    
    start = time.time()
    elapsed = 0
    for i in tqdm.tqdm(range(0, len(files), batch_size)):
        batch = files[i:i + batch_size]
        for file in batch:
            mat_to_mp4(file) ## makes sur that depth file is created
        # main(batch)
        depth_file = file.replace(".mp4", "_depth.mp4")
        depth_batch = [file.replace(".mp4", "_depth.mp4") for file in batch]
        
        main(depth_batch)
        end = time.time()
        elapsed += end - start
        estimated = (end - start) * (len(files) - i - batch_size)
        start = end
    v2e_quit()
