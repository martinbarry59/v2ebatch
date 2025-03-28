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
import h5py
import torchvision.transforms as transforms
from v2ecore.v2e_utils import save_array_to_h5
import v2ecore.model as model
import logging
from multiprocessing import Pool
import warnings
from torch.utils.data import Dataset
import torch.nn.functional as F
warnings.filterwarnings(
    "ignore", category=UserWarning,
    module="torch.nn.functional")
# https://github.com/fastai/fastai/issues/2370

logger = logging.getLogger(__name__)

class FramesListDataset(Dataset):
    def __init__(self, vid_path, ori_dim, transform=None):
        """
        Args:
            file_list (list): List of .npy file paths.
            frame_size (tuple): Desired frame size.
            transform (callable, optional): Optional transform to apply.
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.origDim = ori_dim
        self.dim = (int(self.origDim[0] / 32) * 32,
                    int(self.origDim[1] / 32) * 32)
        with h5py.File(vid_path, 'r') as f:
            self.images =  f['video'][:]
            self.images = F.interpolate(torch.from_numpy(self.images).float().unsqueeze(1).to(self.device), size=self.dim, mode='bilinear', align_corners=False)
            self.images =self.images/255 - 0.48
            
        
        
    def __len__(self):
        return len(self.images) - 1

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
        
        return [self.images[index], self.images[index + 1]]
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

        self.model_loaded = False


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
            shuffle=False,

        )
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

        warper = model.backWarp(dim[1],
                                dim[0],
                                self.device)
        warper = warper.to(self.device)

        # dict1 = torch.load(self.checkpoint, map_location='cpu')
        # fails intermittently on windows

        dict1 = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        interpolator.load_state_dict(dict1['state_dictAT'])
        flow_estimator.load_state_dict(dict1['state_dictFC'])

        return flow_estimator, warper, interpolator

  
    def set_video_loaders(self, videos_path):
        idx = 0
        video_loaders = []
        ori_dims = []
        
        for vid_indx in range(len(self.vids)):
            frame_size = (self.vids[vid_indx].output_height, self.vids[vid_indx].output_width)
            path = os.path.join(videos_path,f"video_{str(vid_indx).zfill(4)}.h5")
            video_loader, dim, ori_dim = self.__load_data(path, frame_size)
            video_loaders.append(video_loader)
            ori_dims.append(ori_dim)
            
            idx += 1

        return video_loaders, ori_dims, dim
    def process_videos_tensor(self, Ft_p, ori_dims, upsampling_factor):
        """
        Efficiently processes video frames, resizing and keeping everything in tensor format.

        Parameters:
        - Ft_p: Tensor of shape (num_videos, num_batch_frames, C, H, W)
        - ori_dims: List of (H, W) tuples specifying original dimensions per video.
        - upsampling_factor: Factor to adjust output indexing.

        Returns:
        - output_tensor: Tensor of resized frames (num_videos, num_batch_frames, C, new_H, new_W).
        """
 

        
        # Stack back into a tensor
        output_tensor = F.interpolate(Ft_p.to(self.device), size= ori_dims[0], mode='bilinear', align_corners=False)
        # Compute output indices (vectorized version)


        return output_tensor.squeeze(1)
 
    def interpolate_batch(self, source_frame_paths):
        """
        Interpolate frames for multiple videos in parallel.
        
        Parameters:
        - source_frame_paths: list of paths, each containing frames from a different video.
        - output_folder: str, directory where interpolated frames will be saved.
        - frame_size: tuple (width, height)
        """
        
        # videos_grouped = group_videos_by_name(source_frame_paths)
        

        video_loaders, ori_dims, dim = self.set_video_loaders(source_frame_paths)
        num_videos = len(video_loaders)
        
        if not self.model_loaded:
            self.flow_estimator, self.warper, self.interpolator = self.__model(dim)
            self.model_loaded = True
        extended_frames = (video_loaders[0].dataset.images.shape[0] -1) * self.upsampling_factor
        all_frames_slomo = torch.zeros((num_videos, extended_frames, ori_dims[0][0], ori_dims[0][1]), device=self.device, dtype=torch.uint8)
        inputFrameCounter = 0
        outputFrameCounter = 0
        

        with torch.no_grad():
            interpTimes=None
            import time
            start = time.time()
            for batch_data in  zip(*video_loaders):
                print(f"Time taken for one batch: {time.time()-start}")
                I0_batch = torch.stack([data[0] for data in batch_data])
                I1_batch = torch.stack([data[1] for data in batch_data])
                
                
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
                    
                    # Ft_p = Ft_p.view(num_videos, num_batch_frames, *Ft_p.shape[1:])
                    # Save frames in parallel
                    
                    output_tensors = self.process_videos_tensor(Ft_p, ori_dims, self.upsampling_factor)
                    output_tensors = output_tensors.view(num_videos, num_batch_frames, *output_tensors.shape[1:])
                    outputFrameIdx=outputFrameCounter + torch.arange(num_batch_frames) * self.upsampling_factor + intermediate_idx
                    all_frames_slomo[:, outputFrameIdx , :, :] = ((output_tensors + 0.48)* 255 ).clamp(0, 255).to(torch.uint8)
                    # all_frames_slomo[:, outputFrameIdx , :, :] = ((output_tensors ).clamp(0, 1)* 255).to(torch.uint8)
                    # for vid_idx in range(num_videos):
                    #     for batch_idx in range(num_batch_frames):
                    #         print(Ft_p[vid_idx, batch_idx])
                    #         img = self.to_image(Ft_p[vid_idx, batch_idx].cpu().detach())
                    #         print(img.size )
                    #         exit()
                    #         img_resize = img.resize(ori_dims[vid_idx], Image.BILINEAR)
                    #         output_idx = outputFrameCounter + self.upsampling_factor * batch_idx + intermediate_idx
                    #         save_path = os.path.join( tmp_output_folder, f'video_{str(vid_idx).zfill(4)}_frame_{str(output_idx).zfill(8)}.png')
                    #         img_resize.save(save_path)
                
                inputFrameCounter += num_batch_frames # batch_size-1 because we repeat frame1 as frame0
                outputFrameCounter += numOutputFramesThisBatch # batch_size-1 because we repeat frame1 as frame0
        to_save = all_frames_slomo.cpu().numpy()

        import cv2
        for img in to_save[0]:
            cv2.imshow("img", img)
            cv2.waitKey(100)
        save_args = []
        for idx, vid in enumerate(self.vids):
            path = vid.vid_slomo  # a simple string path
            array = to_save[idx]
            save_args.append((path, array))

        with Pool(processes=16) as pool:
            pool.map(save_array_to_h5, save_args)
        # for vid in self.vids:
        #     with h5py.File(vid.vid_slomo, "w") as f:
        #         f.create_dataset("vids", data=to_save[idx], compression="gzip")

        #     idx += 1
        print(f"time taken after saving: {time.time()-start}")
        return all_frames_slomo, interpTimes, self.upsampling_factor
    

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
