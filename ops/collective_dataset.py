import torch
from torch.utils import data
import torchvision.models as models
import torchvision.transforms as transforms

import random
from PIL import Image
from scipy.stats import mode
import numpy as np

from collections import Counter


FRAMES_NUM={1: 302, 2: 347, 3: 194, 4: 257, 5: 536, 6: 401, 7: 968, 8: 221, 9: 356, 10: 302, 
            11: 1813, 12: 1084, 13: 851, 14: 723, 15: 464, 16: 1021, 17: 905, 18: 600, 19: 203, 20: 342, 
            21: 650, 22: 361, 23: 311, 24: 321, 25: 617, 26: 734, 27: 1804, 28: 470, 29: 635, 30: 356, 
            31: 690, 32: 194, 33: 193, 34: 395, 35: 707, 36: 914, 37: 1049, 38: 653, 39: 518, 40: 401, 
            41: 707, 42: 420, 43: 410, 44: 356}

 
FRAMES_SIZE={1: (480, 720), 2: (480, 720), 3: (480, 720), 4: (480, 720), 5: (480, 720), 6: (480, 720), 7: (480, 720), 8: (480, 720), 9: (480, 720), 10: (480, 720), 
             11: (480, 720), 12: (480, 720), 13: (480, 720), 14: (480, 720), 15: (450, 800), 16: (480, 720), 17: (480, 720), 18: (480, 720), 19: (480, 720), 20: (450, 800), 
             21: (450, 800), 22: (450, 800), 23: (450, 800), 24: (450, 800), 25: (480, 720), 26: (480, 720), 27: (480, 720), 28: (480, 720), 29: (480, 720), 30: (480, 720), 
             31: (480, 720), 32: (480, 720), 33: (480, 720), 34: (480, 720), 35: (480, 720), 36: (480, 720), 37: (480, 720), 38: (480, 720), 39: (480, 720), 40: (480, 720), 
             41: (480, 720), 42: (480, 720), 43: (480, 720), 44: (480, 720)}


ACTIONS=['NA','Crossing','Waiting','Queueing','Walking','Talking']
ACTIVITIES=['Crossing','Waiting','Queueing','Walking','Talking']


ACTIONS_ID={a:i for i,a in enumerate(ACTIONS)}
ACTIVITIES_ID={a:i for i,a in enumerate(ACTIVITIES)}


def collective_read_annotations(path, sid, num_frames=10):
    annotations={}
    path = path + '/seq%02d/annotations.txt' % sid
    
    with open(path, mode='r') as f:
        frame_id=None
        group_activity=None
        actions=[]
        for l in f.readlines():
            values=l[:-1].split('	')
            if int(values[0]) != frame_id:
                if frame_id != None and frame_id + num_frames <= FRAMES_NUM[sid]: # and frame_id%10==1 and frame_id+9<=FRAMES_NUM[sid]:
                    counter = Counter(actions).most_common(2)
                    group_activity= counter[0][0]-1 if counter[0][0]!=0 else counter[1][0]-1
                    annotations[frame_id]={
                        'frame_id':frame_id,
                        'group_activity':group_activity,
                    }
                    
                frame_id=int(values[0])
                group_activity=None
                actions=[]
                
            actions.append(int(values[5])-1)
        
        if frame_id!=None and frame_id + num_frames <= FRAMES_NUM[sid]: # and frame_id%10==1 and frame_id+9<=FRAMES_NUM[sid]:
            counter = Counter(actions).most_common(2)
            group_activity= counter[0][0]-1 if counter[0][0]!=0 else counter[1][0]-1
            annotations[frame_id]={
                'frame_id':frame_id,
                'group_activity':group_activity,
            }
    return annotations
            
def collective_read_dataset(path, seqs):
    data = {}
    for sid in seqs:
        data[sid] = collective_read_annotations(path,sid)
    return data

def collective_all_frames(anns):
    return [(s,f)  for s in anns for f in anns[s] ]

class CollectiveDataset(data.Dataset):
    """
    Characterize collective dataset for pytorch
    """
    def __init__(self, anns, frames, images_path, image_size, num_frames=10, is_training=True, is_finetune=False):
        self.anns = anns
        self.frames = frames
        self.images_path = images_path
        self.image_size = image_size
        
        self.num_frames = num_frames
        
        self.is_training = is_training
        self.is_finetune = is_finetune
    
    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.frames)
    
    def __getitem__(self, index):
        """
        Generate one sample of the dataset
        """
        
        select_frames = self.get_frames(self.frames[index])
        
        sample = self.load_samples_sequence(select_frames)
        
        return sample
    
    def get_frames(self, frame):
        
        sid, src_fid = frame
        
        if self.is_finetune:
            if self.is_training:
                fid=random.randint(src_fid, src_fid+self.num_frames-1)
                return [(sid, src_fid, fid)]
        
            else:
                return [(sid, src_fid, fid) 
                        for fid in range(src_fid, src_fid+self.num_frames)]
            
        else:
            if self.is_training:
                sample_frames = list(range(src_fid, src_fid + self.num_frames))
                return [(sid, src_fid, fid) for fid in sample_frames]

            else:
                sample_frames=[ src_fid, src_fid+3, src_fid+6, src_fid+1, src_fid+4, src_fid+7, src_fid+2, src_fid+5, src_fid+8 ]
                return [(sid, src_fid, fid) for fid in sample_frames]
    
    
    def load_samples_sequence(self, select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        # OH, OW=self.feature_size
        
        images = []
        activities = []
    
        
        for i, (sid, src_fid, fid) in enumerate(select_frames):

            img = Image.open(self.images_path + '/seq%02d/frame%04d.jpg'%(sid,fid))

            img=transforms.functional.resize(img,self.image_size)
            img=np.array(img)

            # H,W,3 -> 3,H,W
            img=img.transpose(2,0,1)
            images.append(img)
            
            
            activities.append(self.anns[sid][src_fid]['group_activity'])
        
        
        images = np.stack(images)
        activities = np.array(mode(activities)[0], dtype=np.int32)
        
        #convert to pytorch tensor
        images=torch.from_numpy(images).float()
        activities=torch.from_numpy(activities).long()
        
        return images, activities