#Standard Machine Learning Libraries
from json import load
from matplotlib.style import available
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim
import torchvision.models as models
import torch.nn.functional as F

#Additional Helper libraries for data processing and visualization
from PIL import Image
import matplotlib.pyplot as plt
import copy

#mps Metal Processing Systems only for M1 Mac replace mps with gpu if available
device = torch.device("mps" if available else "cpu")

img_size = 512

loader = transforms.Compose((
    transforms.Resize(img_size),
    transforms.ToTensor()
))

def image_processing(img):
    img = Image.open(img)

    img = loader(img).unsqueeze(0)
    return img.to(device, torch.float)

style_img = image_processing("images/style/sand_well_sized.jpg")
content_img = image_processing("images/input/naruto_very_well_sized.jpg")

assert style_img.size() == content_img.size() and style_img.shape == content_img.shape

unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(3) # pause a bit so that plots are updated


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')
 

class Content_Loss(nn.Module):
    def __init__(self, target):
        super(Content_Loss, self).__init__()

        #no computation requires back prop therefore we can detach the tensor from computational graph
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)

        return input

#cannot use simple transpose methods such as A @ A.T as used in numpy because it is a 4-D matrix 
def calc_gram_matrix(input):
    a, b, c, d = input.size()

    new_matrix = input.view(a * b, c * d)

    #calculate gram matrix of the new reshaped matrix
    G = torch.mm(new_matrix, new_matrix.T)

    #normalize by dividing by the number of elements
    return G.div(a * b * c * d)


class Style_Loss(nn.Module):
    def __init__(self, target):
        super(Style_Loss, self).__init__()

        self.target = calc_gram_matrix(target).detach()

    def forward(self, input):
        self.loss = F.mse_loss(calc_gram_matrix(input), self.target)

        return input


model = models.vgg19(pretrained=True).features.to(device).eval()

#VGG models prefered values 
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()

        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, input):
        
        return (input - self.mean) / self.std


# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = Content_Loss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = Style_Loss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], Content_Loss) or isinstance(model[i], Style_Loss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

input_img = content_img.clone()

plt.figure()
imshow(input_img, title="Content Image")

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)

    model.requires_grad_(False)

    optimizer = torch.optim.LBFGS([input_img])

    run = [0]

    while run[0] <= num_steps:

        def closure():

            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss

            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight

            content_score *= content_weight

            loss = content_score + style_score

            loss.backward()

            run[0] += 1

            if run[0]%50 == 0:
                print("run no. {}".format(run[0]))
                print("Style Loss {:4f}, Content Loss {:4f}".format(
                    style_score.item(), content_score.item()
                ))

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

output = run_style_transfer(model, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()

torchvision.utils.save_image(output, "Sand Naruto.jpg")