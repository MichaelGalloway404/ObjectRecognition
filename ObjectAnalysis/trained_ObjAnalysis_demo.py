import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

'''
This is a demo of a model I trained and presented using the resnet-9 architecture for a research course
at Eastern Oregon University in Fall 2024
The model can distinguish between planes, cars, birds, cats, deer, dogs, frogs, horses, humans, or ships.
It captures an image from the users webcam and classifies it, the user can hold up a picture or object.
Author: Michael Galloway
'''
DIR_PATH = os.path.dirname(__file__)
IMAGE_PATH = DIR_PATH + '\\test\\test_imgs\\test_img.png'
CLASSES = ['Plane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Person','Ship','Unknown']
MODEL_PATH = DIR_PATH + '\\model_202506241202.pth'

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
# convert image in subfolder of ./test to tenser and Normalize it, thats what the model was trained on 
valid_data_set = ImageFolder(DIR_PATH + '\\test', valid_tfms)

# simple convolutional block
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
            
    if pool: 
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.layer1 = conv_block(in_channels, 64)
        self.layer2 = conv_block(64, 128, pool=True)

        self.residual_block1 = nn.Sequential(
            conv_block(128, 128),
            conv_block(128, 128)
        )

        self.layer3 = conv_block(128, 256, pool=True)
        self.layer4 = conv_block(256, 512, pool=True)

        self.residual_block2 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512)
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.layer1(xb)
        out = self.layer2(out)
        out = self.residual_block1(out) + out
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.residual_block2(out) + out
        out = self.classifier(out)
        return out

# LOAD OUR TRAINED MODEL
model = ResNet9(3, 10)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, predictions = torch.max(yb, dim=1)
    if(_[0].item() <= 5):
        return CLASSES[10]
    return CLASSES[predictions[0].item()]

def show_png_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Can't find camera.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rect_width, rect_height = 300, 300
    rect_x = (frame_width - rect_width) // 2
    rect_y = (frame_height - rect_height) // 2

    print("\nPress 'SPACE' to capture an image or 'q' to exit.\n")
    print("Show me a plane, car, bird, cat, deer, dog, frog, horse, human, or ship.")
    print("Place image or object in the square.")

    caption = "Press 'space-bar' to analyse or 'q' to exit."
    caption2 = 'Show me a plane, car, bird, cat, deer, dog, frog, horse, human, or ship.'
    caption3 = "Place image or object in the square."

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        origin = (10, 20)
        fontScale = .5
        color = (255, 0, 0)
        thickness = 1
        frame = cv2.putText(frame, caption, origin, font, fontScale, color, thickness, cv2.LINE_AA)
        origin = (10, 38)
        frame = cv2.putText(frame, caption2, origin, font, fontScale, color, thickness, cv2.LINE_AA)
        origin = (10, 55)
        color = (0, 0, 255)
        frame = cv2.putText(frame, caption3, origin, font, fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow("Show me a plane, car, bird, cat, deer, dog, frog, horse, human, or ship", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            cropped_image = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]
            cv2.imwrite(IMAGE_PATH, cropped_image)
            image = Image.open(IMAGE_PATH)

            print("Image we see")
            show_png_image(IMAGE_PATH)

            resized_image = image.resize((32, 32))
            resized_image.save(IMAGE_PATH)

            img,_ = valid_data_set[0]

            print("Image Model sees")
            show_png_image(IMAGE_PATH)

            prediction = predict_image(img, model)
            print('Prediction:', prediction)

            print("\nShow me a plane, car, bird, cat, deer, dog, frog, horse, human, or ship.")
            print("Place image or object in the square.\n")

        elif key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
