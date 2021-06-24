import numpy as np
import torch
from torch import optim, nn
from torchvision import transforms, datasets, models
from collections import OrderedDict
from datapro import process_image

pre_models = {"vgg16": 25088, "densenet121": 1024}


def network(structure='vgg16', hidden_units=[2048, 256], lr=0.001, device='gpu'):
    """ Function to define the structure of the network and the classifier
    Args:
        structure: pretrained model to use
        hidden_units: hidden layers to use
        lr: learning rate
        device: device to use for processing
    Returns:
        model: structured model
        criterion: model criterion
        optimizer: optimizes the model
    """

    # Submit porcessing to specified device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)

    # Define a new, untrained feed-forward network as a classifier, using ReLU activations
    for param in model.parameters():
        param.requires_grad = False

    # Define classifier for the model, criterion and optimizer
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(pre_models[structure], hidden_units[0])),
                                ('relu', nn.ReLU()),
                                ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
                                ('relu', nn.ReLU()),
                                ('fc3', nn.Linear(hidden_units[1], 102)),
                                ('output', nn.LogSoftmax(dim=1))
                            ]))
    

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()   

    return model, criterion, optimizer


def validation(model, validloader, criterion,gpu='gpu'):
    """Function that checks the performance of the model on validation set
    Arg:
        model: trained model
        validloader: contains images and labels for validation
        optimizer: optimizes the process
    Returns:
        valid_loss(float): loss after validation process
        accuracy(float): performance of the model
    """
    valid_loss = 0
    accuracy = 0

    for images, labels in validloader:
        if torch.cuda.is_available() and gpu == 'gpu':
            images, labels = images.to('cuda'), labels.to('cuda')
        else:
            images, labels = images.to('cpu'), labels.to('cpu')

        # Perform forward pass through network
        output_probs = model.forward(images)
        valid_loss += criterion(output_probs, labels).item()

        # Calculate accuracy
        probs = torch.exp(output_probs)
        equality = (labels.data == probs.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

        return valid_loss, accuracy


def predict(image_path, model, topk=3,gpu='gpu'):
    """ Function predicts the class (or classes) of an image using a trained deep learning model.
    Arg:
        image_path: path to image
        model: trained model
        topk
    Returns:
        only_probs: probabilities of predicted images
        top_classes: classes of the image
    """
    if torch.cuda.is_available() and gpu == 'gpu':
        model.to('cuda')
    else:
        model.to('cpu')
    model.eval()

    image = process_image(image_path)
    image = torch.from_numpy(np.array([image])).float()

    with torch.no_grad():
        if torch.cuda.is_available() and gpu == 'gpu':
            output = model.forward(image.cuda())
        else:
            output = model.forward(image.cpu())

    probability = torch.exp(output).data
    probs = probability.topk(topk)
    only_probs = probability.topk(topk)[0][0]

    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [int(index_to_class[each]) for each in np.array(probs[1][0])]

    return only_probs, top_classes
