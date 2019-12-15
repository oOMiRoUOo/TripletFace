import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import os

from triplettorch import HardNegativeTripletMiner
from TripletFace.tripletface.core.dataset import ImageFolder
from TripletFace.tripletface.core.model import Encoder
from torch.utils.data import DataLoader
from triplettorch import TripletDataset
from torchvision import transforms
from sklearn.manifold import TSNE
from torch.optim import Adam
from tqdm import tqdm

model = Encoder( 64 )
weights = torch.load( "model/model.pt" )['model']
model.load_state_dict( weights )
jit_model = torch.jit.trace( model, torch.rand(64, 3, 7, 7) )

torch.jit.save( jit_model, "jit_deliver.pt" )