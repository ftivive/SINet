import torch
from model.model import SINet
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageDraw
from PIL import Image
from PIL import ImageFont
import cv2
import argparse
from torchvision.transforms import functional
import torchvision.transforms as T
from timeit import default_timer as timer

def estimate_crowd(in_img, model, outname, device):
    """
    Estimate the crowd density in a single image
    :param in_img: path of input image
    :param model: trained SINet model
    """
    # 1. Read and pre-process input image
    img         = Image.open(in_img).convert('RGB')   
    W, H        = img.size    
    new_H       = round(H / 32) * 32
    new_W       = round(W / 32) * 32
    img_resized = img.resize((new_W, new_H), Image.BILINEAR)
    img_tensor  = T.ToTensor()(img_resized).to(device, dtype=torch.float).unsqueeze(dim=0)
    img_tensor  = functional.normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    

    # 3. Perform a forward pass
    start = timer()
    model.eval()
    with torch.no_grad():    
        output, _ = model(img_tensor)
        output    = output[0].detach().numpy()
        
    
    end = timer()
    print('Inference time: {:.2f} sec'.format(end - start))

    # 5. Visualize the detections
    count  = output.sum().item()
    output = np.array(output).squeeze()
    output = (output / output.max() * 255).astype('uint8')
    output = cv2.resize(output, (W, H), interpolation = cv2.INTER_LINEAR)   
     
    output = Image.fromarray(output, mode ='L')
    # Call draw Method to add 2D graphics in an image
    d1     = ImageDraw.Draw(output)
    # Add Text to an image
    d1.rectangle(((0,0), (150,40)), fill = "white")
    font = ImageFont.truetype("arial.ttf", 25)
    d1.text((5, 5), 'COUNT: ' + str(np.floor(count)), font = font,  fill= "black") #(0, 255, 255,  255))    
    # Display result
    plt.figure()
    plt.subplot(1,2,1)
    plt.axis('off')        
    plt.imshow(img_resized)
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(output)    
    # Save result       
    plt.savefig(outname, bbox_inches='tight', dpi=1200)
    plt.show()          



if __name__ == "__main__":

    # 1. Parse the command arguments
    args = argparse.ArgumentParser(description='Test a trained SINet for crowd estimation')
    args.add_argument('-i', '--input',  default='./images/test_image.jpg', type=str,
                      help='Path of input image file')
    args.add_argument('-o', '--output', default='./images/test_image-output.jpg', type=str,
                      help='Path of the output file')
    args.add_argument('-w', '--weights', default='./checkpoints/checkpoint_best.pth', type=str,
                      help='Path of the trained weights')
    
    cmd_args = args.parse_args()

    assert cmd_args.input is not None, "Please specify the input"
    

    # 2. Load a trained SINet
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trained_model = SINet(32, 1).to(device)
    checkpoint    = torch.load(cmd_args.weights, map_location=torch.device(device))
    trained_model.load_state_dict(checkpoint)   
    print('Loading the trained SINet model: ', cmd_args.weights)

    # 3. Perform detection
    if os.path.isfile(cmd_args.input):  # if input is a file path
        estimate_crowd(cmd_args.input, trained_model, cmd_args.output, device)
    else:
        print('Please check the input. It must be a file path')

