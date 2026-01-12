import numpy as np
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Augmentor:
    def __init__(self, crop_size, do_flip=True):
        # spatial augmentation params
        self.crop_size = crop_size
        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        
    def transform(self, voxel, depth, valid, img):
        ht, wd = voxel.shape[:2]        

        margin_y = 65 #downside
        margin_x = 35 #leftside

        y0 = np.random.randint(0, voxel.shape[0] - self.crop_size[0] - margin_y)
        x0 = np.random.randint(margin_x, voxel.shape[1] - self.crop_size[1])

        y0 = np.clip(y0, 0, voxel.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, voxel.shape[1] - self.crop_size[1])
        
        voxel = voxel[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depth = depth[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img = img[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                voxel = voxel[:, ::-1]
                depth = depth[:, ::-1]
                valid = valid[:, ::-1]
                img = img[:, ::-1]
        
            if np.random.rand() < self.v_flip_prob: # v-flip
                voxel = voxel[::-1, :]
                depth = depth[::-1, :]
                valid = valid[::-1, :]
                img = img[::-1, :]

        return voxel, depth, valid, img
    
    def __call__(self, voxel, depth, valid, img):
        voxel, depth, valid, img = self.transform(voxel, depth, valid, img)
        voxel = np.ascontiguousarray(voxel)
        depth = np.ascontiguousarray(depth)
        valid = np.ascontiguousarray(valid)  
        img = np.ascontiguousarray(img)  
        
        return voxel, depth, valid, img
                        