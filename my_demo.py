import os
import cv2
import yaml
import scipy.io
import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import glob
import torch
import torch.nn as nn
from torchvision import io
from torchvision import datasets
from model import ft_net

# #Choose cpu or gpu device
device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
 

def load_model():
    config_path = os.path.join('./model/ft_ResNet50/opts.yaml')
    with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'
    save_path = os.path.join('./model/ft_ResNet50/net_59.pth')
    model = ft_net(751)
    model.load_state_dict(torch.load(save_path))
    model.classifier.classifier = nn.Sequential()
    model = model.eval()
    return model.to(device)


def load_gallery_imgs_by_path(gallery_path):
    paths = glob.glob(gallery_path+"/*.jpg")
    gallery_imgs = []
    for img_pth in paths:
        img = cv2.imread(img_pth,cv2.IMREAD_COLOR)
        gallery_imgs.append(img)
    return gallery_imgs

def preprocess(img):
    #Preprocess the input image:
    #size
    img = cv2.resize(img,(224,224))
    #color format
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = np.float32(img)/255.0
    #imagenet normalization
    img[:,:,]-=[0.485, 0.456, 0.406]
    img[:,:,]/=[0.229, 0.224, 0.225]
    return img
 
def extract_features(model, img):
    img = preprocess(img)
    img = torch.from_numpy(img.transpose(2,0,1))
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
    fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
    features = outputs.div(fnorm.expand_as(outputs))
    return features

def extract_gallery_features(model, gallery_imgs):
    gallery_features = []
    for img in gallery_imgs:
        features = extract_features(model,img)
        gallery_features.append(features)
    return torch.cat(gallery_features,0)
 
def sort_img(qf, gf):
    #Transpose the query feature
    query = qf.view(-1,1)
    # print(query.shape)

    #Matrix multiplication for similarity
    score = torch.mm(gf,query)    
    score = score.squeeze(1).cpu()
    score = score.numpy()

    #Sort by similarity
    index = np.argsort(-score)  #from large to small
    return index, score[index]
 
def imshow(img, title=None):
    """Imshow for Tensor."""
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
 
def demo(query_path = "./data/MyQuery/download.jpg", gallery_path = "./data/MyGallery"):

    model = load_model()

    #Load query and gallery imgs
    gallery_imgs = load_gallery_imgs_by_path(gallery_path)
    query_img = cv2.imread(query_path)

    #Compute features
    query_features = extract_features(model, query_img)
    gallery_features = extract_gallery_features(model, gallery_imgs)
    
    #Sort by similarity score
    index, score = sort_img(query_features,gallery_features)  
 
    #Visualization
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_img,'query')
    for i in range(10):
        ax = plt.subplot(1,11,i+2)
        ax.axis('off')
        ax.set_title('%d:%.3f'%(i+1,score[i]), color='blue')
        imshow(gallery_imgs[index[i]])

    print("Press any key to exit the demo.")
    input()
 
if __name__=="__main__":
    demo()