import gradio as gr
import fastai
from fastai.vision import *
from fastai.utils.mem import *
from fastai.vision import open_image, load_learner, image, torch
import numpy as np4
import urllib.request
import PIL.Image
from io import BytesIO
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO
import fastai
from fastai.vision import *
from fastai.utils.mem import *
from fastai.vision import open_image, load_learner, image, torch
import numpy as np
import urllib.request
from urllib.request import urlretrieve
import PIL.Image
from io import BytesIO
import torchvision.transforms as T
import torchvision.transforms as tfms

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]
 
    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()

MODEL_URL = "https://www.dropbox.com/s/8zmvqbicbjtbycj/J4photos.pkl?dl=1"
urllib.request.urlretrieve(MODEL_URL, "J4photos.pkl")
path = Path(".")
learn=load_learner(path, 'J4photos.pkl')

urlretrieve("https://s.hdnux.com/photos/01/07/33/71/18726490/5/1200x0.jpg","soccer1.jpg")
urlretrieve("https://cdn.vox-cdn.com/thumbor/4J8EqJBsS2qEQltIBuFOJWSn8dc=/1400x1400/filters:format(jpeg)/cdn.vox-cdn.com/uploads/chorus_asset/file/22466347/1312893179.jpg","soccer2.jpg")
urlretrieve("https://cdn.vox-cdn.com/thumbor/VHa7adj0Oie2Ao12RwKbs40i58s=/0x0:2366x2730/1200x800/filters:focal(1180x774:1558x1152)/cdn.vox-cdn.com/uploads/chorus_image/image/69526697/E5GnQUTWEAEK445.0.jpg","baseball.jpg")
urlretrieve("https://baseball.ca/uploads/images/content/Diodati(1).jpeg","baseball2.jpeg")

sample_images = [["soccer1.jpg"],
                 ["soccer2.jpg"],
                 ["baseball.jpg"],
                 ["baseball2.jpeg"]]

def predict(input):
  img_t = T.ToTensor()(input)
  img_fast = Image(img_t)
  p,img_hr,b = learn.predict(img_fast)
  x = np.minimum(np.maximum(image2np(img_hr.data*255), 0), 255).astype(np.uint8)
  img = PIL.Image.fromarray(x)
  return img 

gr_interface = gr.Interface(fn=predict, inputs=gr.inputs.Image(), outputs="image", title='Legacy-League',examples=sample_images).launch();
