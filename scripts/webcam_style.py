import torch
from torchvision import transforms
from inference.Inferencer import Inferencer
from models.PasticheModel import PasticheModel
from PIL import Image
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_styles = 16
image_size = 512
mean = [0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
model_save_dir = "style16/pastichemodel-FINAL.pth"
transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])

pastichemodel = PasticheModel(num_styles)
inference = Inferencer(pastichemodel,transform,device)
inference.load_model_weights(model_save_dir)


cap = cv2.VideoCapture(0)
choice = 0
choice2 = 1
percentage = 0
while(True):
    ret, frame = cap.read()
    
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    res_im = inference.eval_image(pil_im,choice,choice2,percentage)
    open_cv_image = cv2.cvtColor(np.array(res_im), cv2.COLOR_RGB2BGR)
    
    cv2.imshow('frame',open_cv_image)
    percentage+=0.01
    
    if percentage>1:
        percentage=0
        choice2 = choice
        choice = (choice+1)%16
    print(choice, percentage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()