import os
import cv2
import yaml
import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
 
import torch
import torch.nn as nn
from torchvision import datasets
from model import ft_net
 
device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess(img):
    img = cv2.resize(img,(224,224))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = np.float32(img)/255.0
    img[:,:,]-=[0.485, 0.456, 0.406]
    img[:,:,]/=[0.229, 0.224, 0.225]
    return img
 
def extract_feature(model, img):
    img = preprocess(img)
    img = torch.from_numpy(img.transpose(2,0,1))
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
    fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
    features = outputs.div(fnorm.expand_as(outputs))
    return features
 
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
 
def sort_img(feature, gf, gl):
    query = feature.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    junk_index = np.argwhere(gl==-1)
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index, score[index]
 
def sort_img_faiss(feature, gf, gl):
    import faiss
    # index = faiss.IndexFlatL2(512)
    index = faiss.IndexFlatIP(512)
    index.add(gf.contiguous().cpu().numpy())
    D, I = index.search(feature.cpu().numpy(), 10)
    # junk_index = np.argwhere(gl==-1)
    # mask = np.in1d(I, junk_index, invert=True)
    # I = I[mask]
    return I.squeeze(),D.squeeze()
 
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
 
def demo(query_path = "data/market1501/query/0001_c1s1_001051_00.jpg"):
    model = load_model()
    img = cv2.imread(query_path)
    feature = extract_feature(model, img)
    result = scipy.io.loadmat('pytorch_result.mat')
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_feature = gallery_feature.to(device)
    gallery_label = result['gallery_label'][0]
    try:
        index,score = sort_img_faiss(feature,gallery_feature,gallery_label)
    except:
        index, score = sort_img(feature,gallery_feature,gallery_label)  
    data_dir = 'data/market1501/pytorch'
    image_datasets = datasets.ImageFolder(os.path.join(data_dir,"gallery"))
 
    try:
        fig = plt.figure(figsize=(16,4))
        ax = plt.subplot(1,11,1)
        ax.axis('off')
        imshow(query_path,'query')
        for i in range(10):
            ax = plt.subplot(1,11,i+2)
            ax.axis('off')
            img_path, _ = image_datasets.imgs[index[i]]
            ax.set_title('%d:%.3f'%(i+1,score[i]), color='blue')
            imshow(img_path)
            print(img_path)
    except RuntimeError:
        for i in range(10):
            img_path = image_datasets.imgs[index[i]]
            print(img_path[0])
        print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
    fig.savefig("query.png")
 
if __name__=="__main__":
    demo()
    input()