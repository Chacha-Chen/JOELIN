
import os
import json
import pickle
import datetime
import matplotlib.pyplot as plt
import numpy as np


import logging




def saveToPickleFile(save_object, save_file):
    with open(save_file, "wb") as pickle_out:
        pickle.dump(save_object, pickle_out)

def loadFromPickleFile(pickle_file):
    with open(pickle_file, "rb") as pickle_in:
        return pickle.load(pickle_in)

def saveToJSONFile(save_dict, save_file):
    with open(save_file, 'w') as fp:
        json.dump(save_dict, fp, default=convert)

def loadFromJSONFile(json_file):
    with open(json_file, 'r') as fp:
        return json.load(fp)

def make_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        logging.info("Creating new directory: {}".format(directory))
        os.makedirs(directory)
        
        
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def log_list(l):
    for e in l:
        logging.info(e)
    logging.info("")
    
    
    
def plot_train_loss(loss_trajectory_per_epoch, trajectory_file):
    plt.cla()
    plt.clf()

    fig, ax = plt.subplots()
    x = [epoch * len(loss_trajectory) + j + 1 for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, avg_loss in enumerate(loss_trajectory) ]
    x_ticks = [ "(" + str(epoch + 1) + "," + str(j + 1) + ")" for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, avg_loss in enumerate(loss_trajectory) ]
    full_train_trajectory = [avg_loss for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, avg_loss in enumerate(loss_trajectory)]
    ax.plot(x, full_train_trajectory)

    ax.set(xlabel='Epoch, Step', ylabel='Loss',
            title='Train loss trajectory')
    step_size = 100
    ax.xaxis.set_ticks(range(0, len(x_ticks), step_size), x_ticks[::step_size])
    ax.grid()

    fig.savefig(trajectory_file)
    
def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

# Credit: https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
