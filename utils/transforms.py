from PIL import Image, ImageFilter
import numpy as np
import cv2
import random
from torchvision.transforms import functional
import numpy

class transform_sample(object):
    def __init__(self, rot_angle, scale, crop, stride, gamma, dataset):
        self.insize  = crop[0]
        self.scale   = scale
        self.crop    = crop
        self.stride  = stride
        self.gamma   = gamma
        self.dataset = dataset
        self.ang_val = rot_angle	

    def __call__(self, image, density_hr, density_mr, density_lr, attention):
        # random resize
        height, width = image.size[1], image.size[0]
        sc_fc         = random.uniform(self.scale[0], self.scale[1])
        in_size       = self.insize / sc_fc 
        if self.dataset == 'SHA':
            if height < width:
                short = height
            else:
                short = width
            if (short < in_size):
                sc_fc = sc_fc * (in_size / short)
               
        height     = round(height * sc_fc)
        width      = round(width  * sc_fc)
        image      = image.resize(         (width, height), Image.BILINEAR)
        density_hr = cv2.resize(density_hr, (width, height), interpolation = cv2.INTER_LINEAR) / (sc_fc * sc_fc)
        density_mr = cv2.resize(density_mr, (width, height), interpolation = cv2.INTER_LINEAR) / (sc_fc * sc_fc)
        density_lr = cv2.resize(density_lr, (width, height), interpolation = cv2.INTER_LINEAR) / (sc_fc * sc_fc)
        attention  = cv2.resize(attention,  (width, height), interpolation = cv2.INTER_NEAREST)

        r_val     = random.random()         
        if (r_val > 0.75 and r_val < 0.95):
            density_hr = Image.fromarray(density_hr)	   
            density_mr = Image.fromarray(density_mr)	   
            density_lr = Image.fromarray(density_lr)	  
            attention  = Image.fromarray(attention)	  
            ang_val    = random.randint(self.ang_val[0], self.ang_val[1])
            attention  = attention.rotate(ang_val)				
            image      = image.rotate(ang_val)		
            density_hr = density_hr.rotate(ang_val)				
            density_mr = density_mr.rotate(ang_val)				
            density_lr = density_lr.rotate(ang_val)				
            attention  = numpy.array(attention)	   
            density_hr = numpy.array(density_hr)	  
            density_mr = numpy.array(density_mr)	  
            density_lr = numpy.array(density_lr)	  
	   	   
        # random crop
        h, w      = self.crop[0], self.crop[1]
        #        
        dh        = random.randint(0, height - h)
        dw        = random.randint(0, width  - w)
        
        r_val     = random.random() 
        if r_val < 0.2:
            dw        = 0
            dh        = 0
        elif (r_val > 0.2  and r_val < 0.4) :
            dw        = width - w
            dh        = 0
        elif (r_val > 0.4  and r_val < 0.6) :
            dw        = 0
            dh        = height - h
        elif (r_val > 0.6  and r_val < 0.8) :
            dw        = width -  w
            dh        = height - h
        image      = image.crop((dw, dh, dw + w, dh + h))
        density_hr = density_hr[dh:dh + h, dw:dw + w]
        density_mr = density_mr[dh:dh + h, dw:dw + w]
        density_lr = density_lr[dh:dh + h, dw:dw + w]
        attention  =  attention[dh:dh + h, dw:dw + w]
        
        r_val = random.random() 
        # blur image
        if r_val > 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius = 1))   
        
        r_val = random.random() 
        if (r_val > 0.1 and r_val < 0.4):
            image      = image.transpose(Image.FLIP_LEFT_RIGHT)
            density_hr = density_hr[:, ::-1]
            density_mr = density_mr[:, ::-1]
            density_lr = density_lr[:, ::-1]
            attention  =  attention[:, ::-1]
			
        r_val = random.random() 
        if (r_val > 0.4 and r_val < 0.6):
            gamma = random.uniform(self.gamma[0], self.gamma[1])
            image = functional.adjust_gamma(image, gamma)
			
        r_val = random.random() 
        if self.dataset == 'SHA':
            if (r_val > 0.6 and r_val < 0.8):
                image = functional.to_grayscale(image, num_output_channels = 3)

        density_hr = cv2.resize(density_hr, (density_hr.shape[1] // self.stride, density_hr.shape[0] // self.stride), interpolation=cv2.INTER_LINEAR) * self.stride * self.stride
        density_mr = cv2.resize(density_mr, (density_mr.shape[1] // self.stride, density_mr.shape[0] // self.stride), interpolation=cv2.INTER_LINEAR) * self.stride * self.stride
        density_lr = cv2.resize(density_lr, (density_lr.shape[1] // self.stride, density_lr.shape[0] // self.stride), interpolation=cv2.INTER_LINEAR) * self.stride * self.stride

        attention  = cv2.resize(attention, (attention.shape[1] // self.stride, attention.shape[0] // self.stride), interpolation=cv2.INTER_NEAREST)

        density_hr = np.reshape(density_hr, [1, density_hr.shape[0], density_hr.shape[1]])
        density_mr = np.reshape(density_mr, [1, density_mr.shape[0], density_mr.shape[1]])
        density_lr = np.reshape(density_lr, [1, density_lr.shape[0], density_lr.shape[1]])

        attention  = np.reshape(attention, [1, attention.shape[0], attention.shape[1]])
        
        return image, density_hr, density_mr, density_lr, attention
