import os
import time
import logging
import uuid
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from model import binarize
import tqdm
from torch.utils.tensorboard import SummaryWriter

# Logging utils
def init_logger(logpath, experiment_name="sample", filepath=None, package_files=None, view_excuted_file=False,
        displaying=True, saving=True, debug=False, tqdm=True) :
    logger = logging.getLogger()
    if debug :
        level = logging.DEBUG
    else :
        level = logging.INFO

    logger.setLevel(level)
    st = time.gmtime()
    experiment_name = experiment_name + "-" + "{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}.log".format(st.tm_year,st.tm_mon,st.tm_mday,st.tm_hour,st.tm_min,st.tm_sec)

    if saving :
        info_file_handler = logging.FileHandler(os.path.join(logpath,experiment_name), mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if tqdm :
        tqdm_handler = TqdmLoggingHandler(level=logging.INFO)
        logger.addHandler(tqdm_handler)

    # Time
    start_time = time.strftime("%Y-%m-%d")
    excution_id = str(uuid.uuid4())[:8]
    logger.info("Experiment name : {}".format(experiment_name))
    logger.info("Start tiem : {}".format(start_time))
    logger.info("Execution ID : {}".format(excution_id))

    # For view whole codes
    if view_excuted_file :
        logger.info("="*80)
        logger.info("excuted file : {}".format(filepath))
        logger.info("="*80)
        with open(filepath,"r") as f :
            logger.info(f.read())

        for f in package_files :
            logger.info("package files : {}".format(f))
            with open(f, "r") as package_f :
                logger.info(package_f.read())

    return logger

class TqdmLoggingHandler(logging.Handler) :
    def __init__(self, level=logging.NOTSET) :
        super().__init__(level)

    def emit(self, record) :
        try :
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit) :
            raise
        except:
            self.handleError(record)

# Method utils
def one_hot(x, K) :
    return np.array(x[:,None] == np.arange(K)[None, :], dtype=int)

def accuracy(model, dataset_loader, device, repeat=1, save_adv=None, criterion=None, attack=None, binarize=False) :
    if save_adv is not None :
        writer=SummaryWriter(log_dir=save_adv)
    total_correct = 0
    criterion = criterion or torch.nn.CrossEntropyLoss().to(device)
    total_loss = []
    for i, (x,y) in enumerate(dataset_loader) :
        if attack is not None :
            x_nat = x.detach().clone()
            x = attack.perturb(x.to(device), y.to(device), device=device)
            if repeat != 1 :
                y = torch.cat([y for _ in range(repeat)])
            if save_adv is not None :
                nat_image = torchvision.utils.make_grid(x_nat.cpu(), scale_each=False)
                adv_image = torchvision.utils.make_grid(x.cpu(), scale_each=False)
                writer.add_image("natural_image", nat_image, i)
                writer.add_image("adversarial_image", adv_image, i)

        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y).cpu().detach().numpy()
        total_loss.append(loss)
        predicted_class = torch.argmax(pred.cpu().detach(), dim=1)
        correct = (predicted_class == y.cpu())
        total_correct += torch.sum(correct).item()

    if save_adv is not None :
        writer.close()
    #total_loss = total_loss.mean()
    total_loss = np.mean(total_loss)
    return total_correct / (len(dataset_loader.dataset)*repeat), total_loss

# Visualization utils
def converter(image) :
    convert = transforms.ToPILImage()
    image = torch.tensor(image)
    image = image.resize(image.size(-3),image.size(-2),image.size(-1))
    return convert(image)

def subset_sampler(source, num_image) :
    # source : torchvision.datasets format
    subset_indice = list(torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(source), batch_size=num_image, drop_last=True))[0]
    subset = torch.utils.data.Subset(source,subset_indice)
    return subset

class RunningAverageMeter(object) :
    """Computes and stores the averate and current value"""

    def __init__(self, momentum=0.99) :
        self.momentum = momentum
        self.reset()

    def reset(self) :
        self.val = None
        self.avg = 0

    def update(self, val) :
        if self.val is None :
            self.avg = val
        else :
            self.avg = self.avg * self.momentum + val * (1-self.momentum)
        self.val = val

if __name__ == "__main__" :
    init_logger("logs")
