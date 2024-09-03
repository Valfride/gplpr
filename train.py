from memory_profiler import profile
import yaml
import torch
import torch.nn as nn
import train_funcs
import utils
import random
import argparse
import models
import losses
import smtplib
import datasets
import numpy as np
from tqdm import tqdm

from PIL import Image
from pathlib import Path
from torchvision import transforms
from email.mime.text import MIMEText
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

#Set gpu visibility, for debbug purposes
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Enable anomaly detection in PyTorch autograd. Anomaly detection helps in finding operations that
# are not supported by autograd and can be useful for debugging. It is often used during development
# and debugging phases.
torch.autograd.set_detect_anomaly(True)

# Set the per-process GPU memory fraction to 90%. This means that the GPU will allocate a maximum
# of 90% of its available memory for this process. This can be useful to limit the GPU memory usage
# when running multiple processes or to prevent running out of GPU memory.
torch.cuda.set_per_process_memory_fraction(1.0, 0)

# Clear the GPU memory cache. This releases GPU memory that is no longer in use and can help free up
# memory for other operations. It's particularly useful when working with limited GPU memory.
torch.cuda.empty_cache()

def make_dataloader(spec, tag=''):
    # Create the dataset based on the provided specification
    dataset = datasets.make(spec['dataset'])
    # Create a dataset wrapper based on the provided specification and the previously created dataset
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    
    loader = DataLoader(
        dataset,
        batch_size=spec['batch'],
        shuffle=(tag == 'train'), # Shuffle the data if the tag is 'train'
        num_workers=0, # Number of worker processes for data loading (0 means data is loaded in the main process)
        pin_memory=False, # Whether to use pinned memory (typically used with CUDA, set to False here)
        collate_fn=dataset.collate_fn # A function used to collate (combine) individual samples into batches
    )

    # """Next is only for debugging purpose"""
    # for batch in loader:
    #     a=0
    # Return the DataLoader        
    return loader 

def make_dataloaders():
    # Create data loaders for the training and validation datasets
    # These data loaders are typically created using custom functions (e.g., make_dataloader)
    train_loader = make_dataloader(config['train_dataset'], tag='train')
    val_loader = make_dataloader(config['val_dataset'], tag='val')

    # Return the created data loaders
    return train_loader, val_loader
    
def prepare_training():
    # Check if a training checkpoint is specified in the configuration (resuming training)
    if config.get('resume') is not None:
        # Load the saved checkpoint file
        sv_file = torch.load(config['resume'])
        
        # Create the model using the configuration from the checkpoint and move it to the GPU
        model = models.make(sv_file['model'], load_model=True).cuda()
        
        # Create an optimizer with parameters from the checkpoint and load its state
        optimizer = utils.make_optimizer(model.parameters(), sv_file['optimizer'], load_optimizer=True)
        
        # Create an EarlyStopping object using settings from the checkpoint
        early_stopper = utils.Early_stopping(**sv_file['early_stopping'])
        
        # Get the starting epoch from the checkpoint and set the random number generator state
        epoch_start = sv_file['epoch'] + 1     
        state = sv_file['state']                
        torch.set_rng_state(state)
        
        # Print a message indicating that training is resuming
        print(f'Resuming from epoch {epoch_start}...')
        log(f'Resuming from epoch {epoch_start}...')
        
        # Check if a learning rate scheduler (ReduceLROnPlateau) is specified in the configuration
        if config.get('reduce_on_plateau') is None:
            lr_scheduler = None
        else:
            lr_scheduler = ReduceLROnPlateau(optimizer, **config['reduce_on_plateau'])
        
        # Set the learning rate scheduler's last_epoch to the resumed epoch
        lr_scheduler.last_epoch = epoch_start - 1
       
    # If no checkpoint is specified, start training from scratch
    else:
        print('Training from start...')
        
        # Create the model using the configuration and move it to the GPU
        model = models.make(config['model']).cuda()
        
        # Create an optimizer using the configuration
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        # Create an EarlyStopping object using settings from the configuration
        early_stopper = utils.Early_stopping(**config['early_stopper'])

        # Set the starting epoch to 1
        epoch_start = 1
        
        # Check if a learning rate scheduler (ReduceLROnPlateau) is specified in the configuration
        if config.get('reduce_on_plateau') is None:
            lr_scheduler = None
        else:
            # Create a learning rate scheduler using settings from the configuration
            lr_scheduler = ReduceLROnPlateau(optimizer, **config['reduce_on_plateau'])
            #StepLR(optimizer, step_size=50, gamma=0.8, verbose=True)
            # lr_scheduler = ReduceLROnPlateau(optimizer, **config['reduce_on_plateau'])
            
        # For epochs prior to the starting epoch, step the learning rate scheduler
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
            
    # Log the number of model parameters and model structure
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    log('model: #struct={}'.format(model))
    
    # Return the model, optimizer, starting epoch, learning rate scheduler, and EarlyStopping object
    return model, optimizer, epoch_start, lr_scheduler, early_stopper

def main(config_, save_path):
    # Declare global variables
    global config, log, writer
    config = config_
    
    # Create log and writer for logging training progress
    log, writer = utils.make_log_writer(save_path)
    
    # Create data loaders for training and validation datasets
    train_loader, val_loader = make_dataloaders()
    
    # Initialize the model, optimizer, learning rate scheduler, and early stopper
    model, optimizer, epoch_start, lr_scheduler, early_stopper = prepare_training()
    train = train_funcs.make(config['func_train'])
    validation = train_funcs.make(config['func_val'])
    # Create the loss function for training    
    loss_fn = losses.make(config['loss'])
    
    # Get the number of available GPUs
    n_gpus = torch.cuda.device_count()
    print(n_gpus)
    
    # If multiple GPUs are available, use DataParallel to parallelize model training
    # if n_gpus > 1:
    #     model = nn.parallel.DataParallel(model)
        
    # Get maximum number of epochs and epoch save interval from configuration
    epoch_max = config['epoch_max']
    epoch_save = config.get('epoch_save')
    
    # Create a timer to measure training time
    timer = utils.Timer()  
    confusing_pair = []
    # Loop over epochs for training
    for epoch in range(epoch_start, epoch_max+1):
        # Initialize timer for the current epoch
        print(f"epoch {epoch}/{epoch_max}")
        t_epoch_init = timer._get()
        
        # Prepare logging information for the current epoch
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        
        # Log the learning rate and add it to the writer
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        log_info.append('lr:{}'.format(optimizer.param_groups[0]['lr']))
        
        # Perform training for the current epoch and get the training loss
        train_loss = train(train_loader, model, optimizer, loss_fn, confusing_pair, config) 
        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalar('train_loss', train_loss, epoch)
        
        # Perform validation for the current epoch and get the validation loss
        val_loss, confusing_pair = validation(val_loader, model, loss_fn, confusing_pair, config)             
        log_info.append('val: loss={:.4f}'.format(val_loss))
        writer.add_scalar('val_train_loss', val_loss, epoch)
        
        # Adjust the learning rate using the learning rate scheduler if it's defined
        if lr_scheduler is not None:
            #print(lr_scheduler)
            lr_scheduler.step(val_loss)
            #print('val_loss (Accuracy): ', val_loss)
        
        # Calculate and log elapsed times for the current epoch
        t = timer._get()        
        t_epoch = timer.time_text(t - t_epoch_init )
        t_elapsed = timer.time_text(t)
        log_info.append('{} / {}'.format(t_epoch, t_elapsed))
        
        # Check for early stopping and log the status
        stop, bm = early_stopper.early_stop(val_loss)
        log_info.append('Early stop {} / Best model {}'.format(stop, bm))
        
        # Get the underlying model (without DataParallel) if multiple GPUs are used
        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        
        # Prepare model and optimizer specifications for saving
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        early_stopper_ = vars(early_stopper)
        
        # Get the current random number generator state
        state = torch.get_rng_state()
        
        # Create a dictionary to save the model checkpoint
        sv_file = {
            'model': model_spec, 
            'optimizer': optimizer_spec, 
            'epoch': epoch, 
            'state': state, 
            'early_stopping': early_stopper_
            }
        
        # Save the model checkpoint if it's the best model so far
        if bm:
            torch.save(sv_file, save_path / Path('best_model_'+'Epoch_{}'.format(epoch)+'.pth'))
        else:
            torch.save(sv_file, save_path / Path('epoch-last.pth'))
        
        # Save the model checkpoint if it's an epoch save interval
        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file, save_path / Path('epoch-{}.pth'.format(epoch)))
         

        # Log the training progress for the current epoch
        log(', '.join(log_info))
        writer.flush()
        
        # Check for early stopping and break the loop if early stopping criteria are met
        if stop:
            print('Early stop: {}'.format(stop))
            break



if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--save', default=None)    
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()
    
    
    # Define a function to set random seeds for reproducibility
    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # sets the seed for cpu
        torch.cuda.manual_seed(seed)  # Sets the seed for the current GPU.
        torch.cuda.manual_seed_all(seed)  #  Sets the seed for the all GPU.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # Set a fixed random seed (for reproducibility)
    setup_seed(1996)
    
    # Read the configuration file (usually in YAML format)
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Determine the save name for checkpoints
    save_name = args.save
    if save_name is not None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
        
    # Create a save path for model checkpoints and ensure the directory exists
    save_path = Path('./save') / Path(save_name)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Call the main training function with the configuration and save path
    main(config, save_path)
    
    # Send an email notification about training completion (optional)
    msg = MIMEText("Your training process has completed successfully.")
    msg['Subject'] = config['email']['subject'] + '_' + config['model']['name']
    msg['From'] = config['email']['sender']
    msg['To'] = config['email']['recipient']
    
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(config['email']['sender'], config['email']['passwd'])
    server.sendmail(config['email']['sender'], config['email']['recipient'], msg.as_string())
    server.quit()
