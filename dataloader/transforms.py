import torch
import torchvision
import torchvision.transforms as transforms
import skimage.transform
import numpy as np
from PIL import Image



def image_transforms(mode='train', augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                     do_augmentation=True, transformations=None,  size=(256, 512)):
    if mode == 'train':
        data_transform = torchvision.transforms.Compose([\
            #Resize(),
            # AddNoise('pts_model_cam', do_augmentation),
            ToTensor(),
            #ColorJitter(do_augmentation),
            NormalizeRGB(do_augmentation),
        ])
        return data_transform
    elif mode == 'test':
        data_transform = torchvision.transforms.Compose([
            ToTensor(),
            NormalizeRGB(do_augmentation),
        ])
        return data_transform
    elif mode == 'custom':
        data_transform = torchvision.transforms.Compose(transformations)
        return data_transform
    else:
        print('Wrong mode')

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

class Resize(object):
    def __init__(self, train=True, size=(160, 160)):
        self.train = train
        self.transform = transforms.Resize(size)
        self.size = size

    def __call__(self, sample):
        if self.train:
            rgb_masked = sample['rgb_masked']
            depth_masked = sample['depth_masked']
            print('rgb_masked', rgb_masked.shape)
            print('depth_masked', depth_masked.shape)
            rgb_resized = skimage.transform.rescale(rgb_masked, self.size, order=1, preserve_range=True)
            depth_resized = skimage.transform.rescale(depth_masked, self.size, order=0, preserve_range=True)

            sample['rgb_resized'] = rgb_resized
            sample['depth_resized'] = depth_resized
        return sample

class ResizeImage(object):
    def __init__(self, train=True, size=(256, 512)):
        self.train = train
        self.transform = transforms.Resize(size)

    def __call__(self, sample):
        if self.train:
            rgb = sample['rgb']
            rgb = self.transform(left_image)
            sample['rgb'] = rgb
        return sample


class DoTest(object):
    def __call__(self, sample):
        new_sample = torch.stack((sample, torch.flip(sample, [2])))
        return new_sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        if batch['rgb'] is None:
            return batch
        if not(_is_numpy_image(batch['rgb'])):
            raise TypeError('img should be ndarray. Got {}'.format(type(batch['rgb'])))

        batch['rgb'] = torch.from_numpy(batch['rgb'].transpose((2, 0, 1)).copy()).float()
        batch['rgb_masked'] = torch.from_numpy(batch['rgb_masked'].transpose((2, 0, 1)).copy()).float()
        batch['depth'] = torch.from_numpy(batch['depth'].copy()).float()
        batch['pts_sensor_cam'] = torch.from_numpy(batch['pts_sensor_cam'].copy()).float()
        batch['pose'] = torch.from_numpy(batch['pose'].copy()).float()
        batch['mask_in_bbox'] = torch.from_numpy(batch['mask_in_bbox'].copy()).long()
        batch['bbox'] = torch.from_numpy(batch['bbox']).long()

        batch['model_points']= torch.from_numpy(batch['model_points'].copy()).float()
        batch['raw_model_points']= torch.from_numpy(batch['raw_model_points'].copy()).float()
        batch['pts_model_cam']= torch.from_numpy(batch['pts_model_cam'].copy()).float()
        batch['model_index'] = torch.LongTensor([batch['model_index']])

        batch['gt_3d_vector_field'] = torch.from_numpy(batch['gt_3d_vector_field'].copy()).float()
        batch['gt_2d_vector_field'] = torch.from_numpy(batch['gt_2d_vector_field'].copy()).float()
        batch['pts_farthest_model'] = torch.from_numpy(batch['pts_farthest_model'].copy()).float()
        batch['pts_farthest_cam'] = torch.from_numpy(batch['pts_farthest_cam'].copy()).float()

        #batch['fpfh_sensor']= torch.from_numpy(batch['fpfh_sensor'].copy()).float()
        #batch['fpfh_model']= torch.from_numpy(batch['fpfh_model'].copy()).float()

        batch['seg_mask'] = torch.from_numpy(batch['seg_mask'].copy()).long()

        
        return batch

class NormalizeRGB:
    def __init__(self, do_augmentation=True):
        self.do_augmentation = do_augmentation
        self.normalization = torchvision.transforms.Normalize(
            # mean = [0, 0, 0],
            # std=[255.0, 255.0, 255.0]
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, batch):
        if self.do_augmentation:
            if batch['rgb'] is not None:
                batch['rgb'] = self.normalization(batch['rgb'])
            if batch['rgb_masked'] is not None:
                batch['rgb_masked'] = self.normalization(batch['rgb_masked'])

        return batch

class ColorJitter:
    def __init__(self, do_augmentation, jitter_range=0.2, hue=0.05):
        self.do_augmentation = do_augmentation
        self.jitter_range = jitter_range
        self.hue = hue
        
    def __call__(self, batch):
       if self.do_augmentation:
           brightness = self.jitter_range#float(self.jitter_range * (2 * torch.rand(1) - 1) + 1)
           contrast = self.jitter_range#float(self.jitter_range * (2 * torch.rand(1) - 1) + 1)
           saturation = self.jitter_range#float(self.jitter_range * (2 * torch.rand(1) - 1) + 1)
           color_jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, self.hue)

           rgb = batch['rgb_masked']

           if _is_numpy_image(rgb):
              pil = Image.fromarray(rgb)
              rgb = np.array(color_jitter(pil))
           else:
              rgb = color_jitter(rgb) if rgb is not None else None
           batch['rgb_masked'] = rgb 
       return batch

class AddNoise:
    def __init__(self, key, do_augmentation, noise_range=0.03):
        self.key = key
        self.do_augmentation = do_augmentation
        self.noise_range = noise_range

    def __call__(self, batch):
        if self.do_augmentation:
            add_t = np.array([np.random.uniform(-self.noise_range, \
                self.noise_range) for i in range(3)])
            batch[self.key] = np.add(batch[self.key], add_t)

        return batch
