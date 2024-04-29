import torch.nn as nn
import torchvision.models as models
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import sys


class MyVGG(nn.Module):
    def __init__(self,feature_layers,device):
        '''
        feature_layers:Index of the selected feature layer
        '''
        super(MyVGG,self).__init__()
        self.feature_layers=feature_layers
        self.net=models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:max(feature_layers)+1].to(device)
    
    def forward(self,x):
        '''
        Used to extract features of different scales from the image
        '''
        features=[]
        for i in range(len(self.net)):
            x=self.net[i](x)
            if i in self.feature_layers:
                features.append(x)
        return features
    

class VideoImageComparator:
    def __init__(self):
        self.MSEF = nn.MSELoss()
        self.device = torch.device("cpu")
        self.model = MyVGG(feature_layers=[5, 10, 19, 25],device=self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image):
        #convert to PIL image
        image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image)
        features = [feature.cpu() for feature in features]
        return features

    def compare_features(self, image1, image2s):
        res=sys.maxsize
        for image2 in image2s:
            features1 = self.extract_features(image1)
            features2 = self.extract_features(image2)
            distance= 1-self.compute_cosine_similarity(features1, features2)
            
            # distance = self.compute_MSE(features1, features2)
            if distance < res:
                res=distance
        return res


    def compute_MSE(self, features1,features2):
        losses = [self.MSEF(f1, f2) for f1, f2 in zip(features1, features2)]
        average_loss = torch.mean(torch.stack(losses))
        return average_loss.item()

    def compute_cosine_similarity(self,features1:list,features2:list):
        '''
        returns cosine similarity
        '''
        cosine_similarities = []
        for feature1_map, feature2_map in zip(features1, features2):
            # flatten the feature maps
            vec1 = feature1_map.view(-1)
            vec2 = feature2_map.view(-1)

            # compute cosine similarity
            cosine_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
            cosine_similarities.append(cosine_sim)

        avg_cosine_similarity = torch.mean(torch.stack(cosine_similarities))
        return avg_cosine_similarity.item()
    
    

    def compare_images_ssim(self, image1, image2s):
        
        image2s_resize=[cv2.resize(target_image, (image1.shape[1], image1.shape[0])) for target_image in image2s]
        gray_image2s_resize = [cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY) for color_image in image2s_resize]
        gray_image1=cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        res=-1
        for image2 in gray_image2s_resize:
            s = ssim(gray_image1, image2)
            if s > res:
                res=s
        
        return 1-res
