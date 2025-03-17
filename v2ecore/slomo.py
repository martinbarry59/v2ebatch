"""Super SloMo class for dvs simulator project.
    @author: Zhe He
    @contact: hezhehz@live.cn
    @latest update: 2019-May-27th

    lightly modified based on this implementation: \
        https://github.com/avinashpaliwal/Super-SloMo
"""

import torch
import os
import numpy as np
import cv2
import glob
from tqdm import tqdm

import torchvision.transforms as transforms
from v2ecore.v2e_utils import all_images, group_videos_by_name
from v2ecore.v2e_utils import video_writer, v2e_quit
import v2ecore.dataloader as dataloader
import v2ecore.model as model
from PIL import Image
import logging
import atexit
import warnings
from torch.utils.data import Dataset
warnings.filterwarnings(
    "ignore", category=UserWarning,
    module="torch.nn.functional")
# https://github.com/fastai/fastai/issues/2370

logger = logging.getLogger(__name__)

class FramesListDataset(Dataset):
    def __init__(self, file_list, ori_dim, transform=None):
        """
        Args:
            file_list (list): List of .npy file paths.
            frame_size (tuple): Desired frame size.
            transform (callable, optional): Optional transform to apply.
        """
        self.files = sorted(file_list)
        self.transform = transform
        self.origDim = ori_dim
        #  self.origDim = array.shape[2], array.shape[1]
        self.dim = (int(self.origDim[0] / 32) * 32,
                    int(self.origDim[1] / 32) * 32)
    def __len__(self):
        return len(self.files) - 1
    def __repr__(self):

        """Return printable representations of the class.
            @Return: str.
        """

        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n',
                                              '\n' + ' ' * len(tmp)))
        return fmt_str

    def __getitem__(self, index):

        """Return an item from the dataset.

            @Parameter:
                index: int.
            @Return: List(Tensor, Tensor).
        """

        sample = []

        image_1 = np.load(self.files[index])
        image_2 = np.load(self.files[index+1])
        # Loop over for all frames corresponding to the `index`.
        for image in [image_1, image_2]:
            # Open image using pil.
            image = Image.fromarray(image)
            image = image.resize(self.dim, Image.Resampling.LANCZOS)
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
        return sample
class SuperSloMo(object):
    """Super SloMo class
        @author: Zhe He
        @contact: hezhehz@live.cn
        @latest update: 2019-May-27th
    """

    def __init__(
            self,
            model: str,
            auto_upsample: bool,
            upsampling_factor: object,
            batch_size=1,
            videos=None,
            vid_orig='original.mp4',
            preview=False,
            avi_frame_rate=30):
        """
        init

        Parameters
        ----------
        model: str,
            path of the stored Pytorch checkpoint.
        upsampling_factor: object,
            slow motion factor.
        auto_upsample: bool,
            Use automatic upsampling, but limit minimum to upsampling_factor
        batch_size: int,
            batch size.
        video_path: str or None,
            str path to folder where you want videos of original and
            slomo video to be stored, else None
        vid_orig: str or None,
            name of output original (input) video at slo motion rate,
            needs video_path to be set too
        vid_slomo: str or None,
            name of slomo video file, needs video_path to be set too

            Returns
            ---------------
            None in case of slowdown_factor=int value.
            np.array of deltaTimes as fractions of source frame interval, based on limiting flow to at most 1 pixel per interframe.
        """

        if torch.cuda.is_available():
            self.device = "cuda:0"
            logger.info('CUDA available, running on GPU :-)')
        else:
            self.device = "cpu"
            logger.warning('CUDA not available, will be slow :-(')


        self.checkpoint = model
        self.batch_size = batch_size
        if not auto_upsample and (not isinstance(upsampling_factor, int) or upsampling_factor < 2):
            raise ValueError(
                'upsampling_factor={} but must be an int value>1 when auto_upsample=True'
                .format(upsampling_factor))

        if upsampling_factor is not None and auto_upsample:
            logger.info('Using auto_upsample and upsampling_factor; setting minimum upsampling to {}'.format(upsampling_factor))

        self.upsampling_factor=upsampling_factor
        self.auto_upsample=auto_upsample

        if upsampling_factor>100:
            logger.warning(f'upsampling_factor={upsampling_factor} which is large, upsampling will take a long time; consider using auto_upsample to limit maximum optical to 1 pixel per upsampled frame')

        if self.auto_upsample:
            logger.info('using automatic upsampling mode')
        else:
            logger.info('upsampling by fixed factor of {}'.format(self.upsampling_factor))

        self.vids = videos
        self.preview = preview
        self.preview_resized = False
        self.avi_frame_rate = avi_frame_rate
        
        # initialize the Transform instances.
        self.to_tensor, self.to_image = self.__transform()
        self.ori_writer = None
        self.slomo_writer = None  # will be constructed on first need
        self.numOrigVideoFramesWritten = 0
        self.numSlomoVideoFramesWritten = 0

        atexit.register(self.cleanup)
        self.model_loaded = False

    def cleanup(self):
        if self.ori_writer is not None:
            logger.info(
                'closing original video AVI {} after '
                'writing {} frames'.format(self.vid_orig, self.numOrigVideoFramesWritten))
            self.ori_writer.release()
        if self.slomo_writer is not None:
            logger.info(
                'closing slomo video AVI {} after '
                'writing {} frames'.format(self.vid_slomo, self.numSlomoVideoFramesWritten))
            self.slomo_writer.release()
        cv2.destroyAllWindows()

    def __transform(self):
        """create the Transform instances.

        Returns
        -------
        to_tensor: Pytorch Transform instance.
        to_image: Pytorch Transform instance.
        """
        mean = [0.428]
        std = [1]
        normalize = transforms.Normalize(mean=mean, std=std)
        negmean = [x * -1 for x in mean]
        revNormalize = transforms.Normalize(mean=negmean, std=std)

        if (self.device == "cpu"):
            to_tensor = transforms.Compose([transforms.ToTensor()])
            to_image = transforms.Compose([transforms.ToPILImage()])
        else:
            to_tensor = transforms.Compose([transforms.ToTensor(),
                                            normalize])
            to_image = transforms.Compose([revNormalize,
                                           transforms.ToPILImage()])
        return to_tensor, to_image

    def __load_data(self, source_frame_paths, frame_size):
        """Return a Dataloader instance, which is constructed with \
            APS frames.

        Parameters
        ---------
        images: np.ndarray, [N, W, H]
            input APS frames.
        Returns
        -------
        videoFramesloader: Pytorch Dataloader instance.
        frames.dim: new size.
        frames.origDim: original size.
        """
        #  frames = dataloader.Frames(images, transform=self.to_tensor)
        frames = FramesListDataset(source_frame_paths, frame_size, transform=self.to_tensor)
        
        videoFramesloader = torch.utils.data.DataLoader(
            frames,
            batch_size=self.batch_size,
            shuffle=False)
        return videoFramesloader, frames.dim, frames.origDim

    def __model(self, dim):
        """Initialize the pytorch model

        Parameters
        ---------
        dim: tuple
            size of resized images.

        Returns
        -------
        flow_estimator: nn.Module
        warpper: nn.Module
        interpolator: nn.Module
        """
        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(
                'SuperSloMo model checkpoint ' + str(self.checkpoint) +
                ' does not exist or is not readable')
        logger.info('loading SuperSloMo model from ' + str(self.checkpoint))

        flow_estimator = model.UNet(2, 4)
        flow_estimator.to(self.device)
        for param in flow_estimator.parameters():
            param.requires_grad = False
        interpolator = model.UNet(12, 5)
        interpolator.to(self.device)
        for param in interpolator.parameters():
            param.requires_grad = False

        warper = model.backWarp(dim[0],
                                dim[1],
                                self.device)
        warper = warper.to(self.device)

        # dict1 = torch.load(self.checkpoint, map_location='cpu')
        # fails intermittently on windows

        dict1 = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        interpolator.load_state_dict(dict1['state_dictAT'])
        flow_estimator.load_state_dict(dict1['state_dictFC'])

        return flow_estimator, warper, interpolator

    def interpolate(self, source_frame_path, output_folder, frame_size):
        """Run interpolation. \
            Interpolated frames will be saved in folder self.output_folder.

        Parameters
        ----------
        source_frame_path: path that contains source file
        output_folder:str, folder that stores the interpolated images,
            numbered 1:N*slowdown_factor.
        frame_size: tuple (width, height)


        Frames will include the input frames, i.e.
        if there are 2 input frames and slowdown_factor=10,
        there will be 10 frames written,
        starting with the first input frame, and ending before
        the 2nd input frame.

        If  slowdown factor=2, then the first output frame will be
        the first input frame, and the 2nd output frame will be
        a new synthetic frame halfway to the 2nd frame.

        If the slowdown_factor is 3, then there will
        the first input frame followed by 2 more interframes.

        The output will never include the 2nd input frame.

        i.e. if there are 2 input frames and slowdown_factor=10,
        there will be 10 frames written, frame0 is the first input frame, and frame9 is the 9th interpolated frame.
        Frame1 is *not* included, so that it can be fed as input for the next interpolation.

        Returns
        deltaTimes: np.array,
            Array of delta times relative to src frame intervals. This array must be multiplied by the source frame interval to obtain the times of the frames. There will be a variable number of times depending on auto_upsample and upsampling_factor.
        avg_upsampling_factor: float,
            Average upsampling factor, which can be used to compute the average timestamp resolution.
        """
        if not output_folder:
            raise ValueError(
                'output_folder is None; it must be supplied to store '
                'the interpolated frames')

        ls=os.listdir(source_frame_path)
        nframes=len(ls)
        del ls
        
        if nframes/self.batch_size<2:
            logger.warning(f'only {nframes} input frames with batch_size={self.batch_size}, automatically reducing batch size to provide at least 2 batches')
            while nframes/self.batch_size<2:
                self.batch_size=int(self.batch_size/2)
            logger.info(f'using batch_size={self.batch_size}')
            
        video_frame_loader, dim, ori_dim = self.__load_data(
            source_frame_path, frame_size)
        
        if not self.model_loaded:
            (self.flow_estimator, self.warper,
             self.interpolator) = self.__model(dim)
            self.model_loaded = True

        if self.video_path is not None and self.vid_slomo is not None and \
                self.slomo_writer is None :
            if "depth" in self.vid_slomo:
                self.slomo_writer = video_writer(
                    os.path.join(self.video_path, self.vid_slomo),
                    ori_dim[1],
                    ori_dim[0], frame_rate=self.avi_frame_rate
                )

        numUpsamplingReportsLeft=3 # number of times to report automatic upsampling

        # prepare preview
        if self.preview:
            self.name = os.path.basename(str(__file__))
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)

        outputFrameCounter=0 # counts frames written out (input + interpolated)
        inputFrameCounter=0 # counts source video input frames
        # torch.cuda.empty_cache()
        upsamplingSum=0 #stats
        nUpsamplingSamples=0
        with torch.no_grad():
            
            nImages = len(video_frame_loader)
            logger.info(f'interpolating {len(video_frame_loader)} batches of frames using batch_size={self.batch_size} with auto_upsample={self.auto_upsample} and minimum upsampling_factor={self.upsampling_factor}')
            if nImages<2:
                raise Exception('there are only {} batches in {} and we need at least 2; maybe you need to reduce batch size or increase number of input frames'.format(nImages, source_frame_path))

            interpTimes=None # array to hold times normalized to 1 unit per input frame interval

            unit = ' fr' if self.batch_size == 1 \
                else ' batch of '+str(self.batch_size)+' fr'
            
            for _, (frame0, frame1) in enumerate(
                    tqdm(video_frame_loader, desc='slomo-interp',
                         unit=unit), 0):
                # video_frame_loader delivers self.batch_size batch of frame0 and frame1
                # frame0 is actually frame0, frame1,.... frameN
                # frame1 is actually frame1, frame2, .... frameN+1, where N is batch_size-1
                # that way the slomo computes in parallel the flow from 0->1, 1->2, 2->3... N-1->N
                
                I0 = frame0.to(self.device)
                I1 = frame1.to(self.device)
                # actual number of frames, account for < batch_size
                num_batch_frames = I0.shape[0]

                flowOut = self.flow_estimator(torch.cat((I0, I1), dim=1))
                F_0_1 = flowOut[:, :2, :, :] # flow from 0 to 1
                F_1_0 = flowOut[:, 2:, :, :] # flow from 1 to 0
                # dimensions [batch, flow[vx,vy], loc_x,loc_y]

                if self.preview:
                    start_frame_count = outputFrameCounter

                # compute the upsampling factor
                if self.auto_upsample:
                    # compute automatic sample time from maximum flow magnitude such that
                    #                 #  dt(s)*speed(pix/s)=1pix,
                    #                 #  i.e., dt(s)=1pix/speed(pix/s)
                    # we have no time here, so our flow is computed in pixels of motion between frames
                    # we need to compute speed, so first compute the sum square of x and y vel components
                    vFlat=torch.flatten(flowOut,2,3) # [batch, [v01x, v01y, v10x, v10y] ]
                    vx0=vFlat[:,0,:]
                    vx1=vFlat[:,2,:]
                    vy0=vFlat[:,1,:]
                    vy1=vFlat[:,3,:]
                    sp0=torch.sqrt(vx0*vx0+vy0*vy0)
                    sp1=torch.sqrt(vx1*vx1+vy1*vy1)
                    sp=torch.cat((sp0,sp1),1)
                    maxSpeed= torch.max(torch.max(sp,dim=1)[0]).cpu().item() # this is maximimum movement between frames in pixels dim [batch]
                    # dim=1 gets max over all pixels
                    # [0] gets value of max, rather than idx which would be 1
                    # outer max get max over entire batch
                    # .cpu() moves to cpu to get actual value as float
                    # outer .item() gets first element of 0-dim tensor which is the speed
                    upsampling_factor=int(np.ceil(maxSpeed)) # use ceil to ensure oversampling. compute overall maximum needed upsampling ratio
                    # it is shared over all frames in batch so just use max value for all of them
                    if self.upsampling_factor is not None and self.upsampling_factor>upsampling_factor:
                        upsampling_factor=self.upsampling_factor
                    if numUpsamplingReportsLeft>0:
                        logger.info('upsampled by factor {}'.format(upsampling_factor))
                        numUpsamplingReportsLeft-=1
                    
                else:
                    
                    upsampling_factor=self.upsampling_factor

                if upsampling_factor<2:
                    logger.warning('upsampling_factor was less than 2 (maybe very slow motion caused this); set it to 2')
                    upsampling_factor=2

                nUpsamplingSamples+=1
                upsamplingSum+=upsampling_factor
                # compute normalized frame times where 1 is full interval between frames
                # each src frame increments time by 1 unit, interframes fill between.
                numOutputFramesThisBatch= upsampling_factor*num_batch_frames
                interframeTime = 1/upsampling_factor
                # compute the times of *all* the new frames, covering upsampling_factor * numFramesThisBatch total frames
                # they all share the same upsampling_factor within this batch, hence same interframeTime
                interframeTimes = inputFrameCounter + np.array(range(numOutputFramesThisBatch))*interframeTime
                interframeTimes = interframeTimes.squeeze() # remove trailing , dimension
                if interpTimes is None:
                    interpTimes=interframeTimes
                else:
                    interpTimes=np.concatenate((interpTimes,interframeTimes))

                # Generate intermediate frames using upsampling_factor
                # this part is also done in batch mode
                for intermediateIndex in range(0, upsampling_factor):
                    t = (intermediateIndex + 0.5) / upsampling_factor
                    temp = -t * (1 - t)
                    fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                    F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                    F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                    g_I0_F_t_0 = self.warper(I0, F_t_0)
                    g_I1_F_t_1 = self.warper(I1, F_t_1)

                    intrpOut = self.interpolator(
                        torch.cat(
                            (I0, I1, F_0_1, F_1_0,
                             F_t_1, F_t_0, g_I1_F_t_1,
                             g_I0_F_t_0), dim=1))

                    F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                    F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                    V_t_0 = torch.sigmoid(intrpOut[:, 4:5, :, :])
                    V_t_1 = 1 - V_t_0

                    g_I0_F_t_0_f = self.warper(I0, F_t_0_f)
                    g_I1_F_t_1_f = self.warper(I1, F_t_1_f)

                    wCoeff = [1 - t, t]

                    Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f +
                            wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / \
                           (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                    # Save intermediate frames from this particular upsampling point between src frames
                    for batchIndex in range(num_batch_frames):
                        
                        img = self.to_image(Ft_p[batchIndex].cpu().detach())
                        img_resize = img.resize(ori_dim, Image.BILINEAR)
                        # the output frame index is computed
                        outputFrameIdx=outputFrameCounter + upsampling_factor * batchIndex + intermediateIndex
                        save_path = os.path.join(
                            output_folder,
                            str(outputFrameIdx) + ".png")
                        img_resize.save(save_path)

                # for preview
                if self.preview:
                    stop_frame_count = outputFrameCounter

                    for frame_idx in range(
                            start_frame_count,
                            stop_frame_count + upsampling_factor * (num_batch_frames - 1)):
                        frame_path = os.path.join(
                            output_folder, str(frame_idx) + ".png")
                        frame = cv2.imread(frame_path)
                        cv2.imshow(self.name, frame)
                        if not self.preview_resized:
                            cv2.resizeWindow(self.name, 800, 600)
                            self.preview_resized = True
                        # wait minimally since interp takes time anyhow
                        k=cv2.waitKey(1)
                        if k==27 or k==ord('x'):
                            v2e_quit()
                # Set counter accounting for batching of frames
                inputFrameCounter += num_batch_frames # batch_size-1 because we repeat frame1 as frame0
                outputFrameCounter += numOutputFramesThisBatch # batch_size-1 because we repeat frame1 as frame0

            # write input frames into video
            # don't duplicate each frame if called using rotating buffer
            # of two frames in a row
            if self.ori_writer:
                src_files = sorted(
                    glob.glob("{}".format(source_frame_path) + "/*.npy"))
                
                # write original frames into stop-motion video
                for frame_idx, src_file_path in enumerate(
                        tqdm(src_files, desc='write-orig-avi',
                             unit='fr'), 0):
                    src_frame = np.load(src_file_path)
                    self.ori_writer.write(
                        cv2.cvtColor(src_frame, cv2.COLOR_GRAY2BGR))
                    self.numOrigVideoFramesWritten += 1

            frame_paths = all_images(output_folder)
            
            if self.slomo_writer:
                for path in tqdm(frame_paths,desc='write-slomo-vid',unit='fr'):
                    frame = self.__read_image(path)
                    self.slomo_writer.write(
                        cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
                    self.numSlomoVideoFramesWritten += 1
        nFramesWritten=len(frame_paths)

        nTimePoints=len(interpTimes)
        avgUpsampling=upsamplingSum/nUpsamplingSamples
        logger.info('Wrote {} frames and returning {} frame times.\nAverage upsampling factor={:5.1f}'.format(nFramesWritten,nTimePoints,avgUpsampling))
        return interpTimes, avgUpsampling

    
    def set_slomo_writer(self, grouped_videos_paths):
        idx = 0
        video_loaders = []
        ori_dims = []
        self.slomo_writers = [None] * len(self.vids)
        for vid_indx in range(len(self.vids)):
            frame_size = (self.vids[vid_indx].output_width, self.vids[vid_indx].output_height)
            video_loader, dim, ori_dim = self.__load_data(grouped_videos_paths[vid_indx], frame_size)
            video_loaders.append(video_loader)
            ori_dims.append(ori_dim)
            
            if self.slomo_writer is None :
                # if "depth" in self.vid_slomo:
                    self.slomo_writers[idx] = video_writer(
                        os.path.join(self.vids[vid_indx].output_folder, self.vids[vid_indx].vid_slomo),
                        ori_dim[1],
                        ori_dim[0], frame_rate=self.avi_frame_rate
                    )
            idx += 1
        return video_loaders, ori_dims, dim
    def interpolate_batch(self, source_frame_paths, tmp_output_folder):
        """
        Interpolate frames for multiple videos in parallel.
        
        Parameters:
        - source_frame_paths: list of paths, each containing frames from a different video.
        - output_folder: str, directory where interpolated frames will be saved.
        - frame_size: tuple (width, height)
        """
 
        videos_grouped = group_videos_by_name(source_frame_paths)
        num_videos = len(videos_grouped)
        video_loaders, ori_dims, dim = self.set_slomo_writer(videos_grouped)
        

        
        
        if not self.model_loaded:
            self.flow_estimator, self.warper, self.interpolator = self.__model(dim)
            self.model_loaded = True
        
        inputFrameCounter = 0
        outputFrameCounter = 0
        with torch.no_grad():
            interpTimes=None
            for batch_data in tqdm(zip(*video_loaders), desc='Batch Processing Videos', unit='batch'):
                
                I0_batch = torch.stack([data[0] for data in batch_data]).to(self.device)
                I1_batch = torch.stack([data[1] for data in batch_data]).to(self.device)
                
                num_batch_frames = I0_batch.shape[1]  # batch size within each video
                
                ## put the two batch dim together
                I0_batch_NET = I0_batch.view(-1, *I0_batch.shape[2:])
                I1_batch_NET = I1_batch.view(-1, *I1_batch.shape[2:])


                flow_out = self.flow_estimator(torch.cat((I0_batch_NET, I1_batch_NET), dim=1))

                ## return to the original shape


                F_0_1, F_1_0 = flow_out[:, :2, :, :], flow_out[:, 2:, :, :]
                numOutputFramesThisBatch= self.upsampling_factor*num_batch_frames
                interframeTime = 1/self.upsampling_factor
                interframeTimes = inputFrameCounter + np.array(range(numOutputFramesThisBatch))*interframeTime
                interframeTimes = interframeTimes.squeeze() # remove trailing , dimension
                if interpTimes is None:
                    interpTimes=interframeTimes
                else:
                    interpTimes=np.concatenate((interpTimes,interframeTimes))
                for intermediate_idx in range(self.upsampling_factor):
                    t = (intermediate_idx + 0.5) / self.upsampling_factor
                    temp = -t * (1 - t)
                    f_coeff = [temp, t * t, (1 - t) * (1 - t), temp]
                    
                    F_t_0 = f_coeff[0] * F_0_1 + f_coeff[1] * F_1_0
                    F_t_1 = f_coeff[2] * F_0_1 + f_coeff[3] * F_1_0
                    
                    g_I0_F_t_0 = self.warper(I0_batch_NET, F_t_0)
                    g_I1_F_t_1 = self.warper(I1_batch_NET, F_t_1)
                    
                    intrp_out = self.interpolator(
                        torch.cat((I0_batch_NET, I1_batch_NET, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                    
                    F_t_0_f = intrp_out[ :, :2, :, :] + F_t_0
                    F_t_1_f = intrp_out[ :, 2:4, :, :] + F_t_1
                    V_t_0 = torch.sigmoid(intrp_out[ :, 4:5, :, :])
                    V_t_1 = 1 - V_t_0
                    
                    g_I0_F_t_0_f = self.warper(I0_batch_NET, F_t_0_f)
                    g_I1_F_t_1_f = self.warper(I1_batch_NET, F_t_1_f)
                    
                    Ft_p = (V_t_0 * g_I0_F_t_0_f + V_t_1 * g_I1_F_t_1_f) / (V_t_0 + V_t_1)
                    
                    Ft_p = Ft_p.view(num_videos, num_batch_frames, *Ft_p.shape[1:])
                    # Save frames in parallel
                    for vid_idx in range(num_videos):
                        for batch_idx in range(num_batch_frames):
                            img = self.to_image(Ft_p[vid_idx, batch_idx].cpu().detach())
                            img_resize = img.resize(ori_dims[vid_idx], Image.BILINEAR)
                            output_idx = outputFrameCounter + self.upsampling_factor * batch_idx + intermediate_idx
                            save_path = os.path.join( tmp_output_folder, f'video_{str(vid_idx).zfill(4)}_frame_{str(output_idx).zfill(8)}.png')
                            img_resize.save(save_path)
                inputFrameCounter += num_batch_frames # batch_size-1 because we repeat frame1 as frame0
                outputFrameCounter += numOutputFramesThisBatch # batch_size-1 because we repeat frame1 as frame0
    
                
        for vid_idx in range(num_videos):
            frame_paths = all_images(tmp_output_folder, vid_idx)
            if self.slomo_writers[vid_idx]:
                for path in tqdm(frame_paths,desc='write-slomo-vid',unit='fr'):
                    frame = self.__read_image(path)
                    self.slomo_writers[vid_idx].write(
                        cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
                    self.numSlomoVideoFramesWritten += 1
                self.slomo_writers[vid_idx].release()
        return interpTimes, self.upsampling_factor
    

    @staticmethod
    def __read_image(path):
        """Read image.

        Parameters
        ----------
        path: str
            path of image.

        Return
        ------
            np.ndarray
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img

    def get_interpolated_timestamps(self, ts):
        """ Interpolate the timestamps.

        Parameters
        ----------
        ts: np.array, np.float64,
            timestamps of input frames.

        Returns
        -------
        np.array, np.float64,
            interpolated timestamps.
        """
        new_ts = []
        for i in range(ts.shape[0] - 1):
            start, end = ts[i], ts[i + 1]
            interpolated_ts = np.linspace(
                start,
                end,
                self.upsampling_factor, # TODO deal with auto mode
                endpoint=False) + 0.5 * (end - start) / self.upsampling_factor
            new_ts.append(interpolated_ts)
        new_ts = np.hstack(new_ts)

        return new_ts
