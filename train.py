import torch
import os
import yaml
from torch import nn
from model.model import SINet
from utils.load_data import crowd_dataset
from torch.utils.data import DataLoader
from torch.optim      import Adam
from model.ssim_loss import ms_ssimloss
from utils.utils import train_model, validate_model, setup_cuda, gaussian, create_window

# Turn off the warning of YAML loader
import warnings
warnings.filterwarnings('ignore')

# Main
if __name__ == '__main__':

    # 1. Setup CUDA
    device = setup_cuda()

    # 2. Load the configurations from the yaml file
    config_path = './configs/config.yml'
    with open(config_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    dataset_dir  = cfg['train_params']['dataset_dir']    
    dataset_type = cfg['train_params']['dataset']    
    num_epochs   = cfg['train_params']['num_epochs']
    batch_size   = cfg['train_params']['batch_size']
    num_workers  = cfg['train_params']['num_workers']
    lr_start     = cfg['train_params']['lr_start']  
    image_size   = cfg['train_params']['image_size'] 
    num_acc      = cfg['train_params']['number_gradient_accumulation'] 
    
    # 3. Load the dataset
    train_dataset = crowd_dataset(dataset_dir = dataset_dir, dataset_type = dataset_type, image_size= image_size, is_train = True)
    train_loader  = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory = True)
    valid_dataset = crowd_dataset(dataset_dir = dataset_dir, dataset_type = dataset_type, image_size= image_size, is_train = False)
    valid_loader  = DataLoader(valid_dataset, batch_size = 1, shuffle = False)

    # 4. Specify the model and loss function
    # define sinet network with 32 channel and 1 output channel
    model    = SINet(32, 1).to(device)
    # define loss function
    fsize       = [11, 9, 7, 5, 3]
    gaussfilter = []
    for i in range(len(fsize)):
        gaussfilter.append(create_window(fsize[i], 1, fsize[i]/6).to(device))
    ssimloss = ms_ssimloss(1, gaussfilter, fsize)

    bceloss  = nn.BCELoss(reduction = 'sum').to(device) 
    l1loss   = nn.L1Loss(reduction  = 'sum').to(device) 
	
    # 5. Specify the optimizer
    optimizer = Adam(model.parameters(), lr = lr_start)
	
    # load previous model if necessary
    load_checkpoint = False    
    save_path       = './checkpoints'  # path to save checkpoint    
    # load checkpoint
    if load_checkpoint:
        checkpoint  = torch.load(os.path.join(save_path, 'checkpoint_best.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_mae    = torch.load(os.path.join(save_path, 'checkpoint_best.pth'))['mae']
        start_epoch = checkpoint['epoch'] + 1
    else:
        best_mae    = 1e4
        start_epoch = 1

    # 6. Start training the model    
    for epoch in range(start_epoch, num_epochs):
		
        # 6.1. Train the model over a single epoch
        train_loss, train_mae = train_model(model, device, optimizer, ssimloss, bceloss, l1loss, train_loader, batch_size, num_acc, epoch)
        
        # 6.2. Validate the model
        mae, mse = validate_model(model, device, valid_loader)        
        print('\nEpoch: {:.0f} -- Train loss {:.2f} -- Train MAE {:.2f} -- Test MAE {:.2f} -- Test MSE: {:.2f}'.format(epoch, train_loss, train_mae, mae, mse))

        # 6.3. Save the model if the validation performance is increasing		        
        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'mae': mae, 'mse': mse}
        torch.save(state, os.path.join(save_path, 'checkpoint_latest.pth'))
        if mae < best_mae:
            torch.save(state, os.path.join(save_path, 'checkpoint_best.pth'))         
            torch.save(model.state_dict(), './checkpoints/Best_Model_' + dataset_type + '_epoch_' + str(epoch) + '_MAE_{:.2f}'.format(mae) + '_MSE_{:.2f}'.format(mse) + '.pth')       
            print('\nMAE decreases ({:.2f} --> {:.2f}). Model saved'.format(best_mae, mae))
            best_mae = mae   
            f = open("./checkpoints/Validate_result.txt", 'a')
            f.write('Epoch = %d, Train_loss = %.2f, Train_MAE = %.2f, Test_MAE = %.2f, Test_MSE = %.2f\n'%(epoch, train_loss, train_mae, mae, mse))
            f.close()

        f = open("./checkpoints/Train_result.txt", 'a')
        f.write('Epoch = %d, Train_loss = %.2f, Train_MAE = %.2f, Test_MAE = %.2f, Test_MSE = %.2f\n'%(epoch, train_loss, train_mae, mae, mse))
        f.close()

    
