import cv2
import os
import numpy as np
from torch import nn
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


def extract_two_sub_images(filepath : str, image_number : int, display_images : bool, n_image : int, n_label : int):
    """extracts two subimages from the given image number. Returns both the label and the image

    Parameters
    ----------
    filepath : str
        the location of the folders containing the masks and the images
    image_number : int
        the number (1-30) of the file to be opened
    display_images : bool
        whether to print the images and their associated large image
    Returns
    -------
    (img_1, label_1), (img_2, label_2) : all numpy.array
        The image (array format) and manual label (array format) in grayscale for two sub images
        the label arrays have a size of 67x67. The input arrays are 97x97 as the network structure
        loses the outer pixel dimensions  
    img_g : numpy.array
        The original image in grayscale, with the chosen rechtangles marked
    """

    file_orig = filepath+"\\Original\\"+str(image_number)+".jpg"
    label = filepath + "\\Manual\\"+str(image_number)+"s.jpg"
    #print(os.listdir(filepath))

    # note - we will train on a gray image!
    img = cv2.imread(file_orig)
    img_g = cv2.imread(file_orig,cv2.IMREAD_GRAYSCALE)
    img_label = cv2.imread(label,cv2.IMREAD_GRAYSCALE)

    # Draw in the labels
    # note that the label are where it is white (255), but there is some noise
    # where the mask is value 1, should be removed prior to training
    #img[img_label == 255] = (0,0,255)
    window_name = 'demo'


    # Find the outer extents
    height, width = img_label.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    cnts, _ = cv2.findContours(img_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        min_x = min(min_x,x)
        min_y = min(min_y,y)
        max_x = max(max_x,x+w)
        max_y = max(max_y,y+h)

    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    # paper uses 239x239px images cropped to 67x67. However, the input is 97x97
    n_sub = n_image
    n_lab = n_label

    offset = 55 #px from image edge

    mid1 = (min_x + offset, (min_y+max_y)//2)
    mid2 = (max_x - offset, (min_y+max_y)//2)
    # we take two sub images at opposite sides of the labelled area    
    # we copy here to avoid redrawing a rectangle on the top
    img_cropped_1 = img_g[(mid1[1] - (n_sub-1)//2) : (mid1[1] + (n_sub-1)//2)+1, (mid1[0] - (n_sub-1)//2) : (mid1[0] + (n_sub-1)//2)+1].copy()
    img_cropped_2 = img_g[(mid2[1] - (n_sub-1)//2) : (mid2[1] + (n_sub-1)//2)+1, (mid2[0] - (n_sub-1)//2) : (mid2[0] + (n_sub-1)//2)+1].copy()

    label_cropped_1 = img_label[(mid1[1] - (n_lab-1)//2) : (mid1[1] + (n_lab-1)//2)+1, (mid1[0] - (n_lab-1)//2) : (mid1[0] + (n_lab-1)//2)+1].copy()
    label_cropped_2 = img_label[(mid2[1] - (n_lab-1)//2) : (mid2[1] + (n_lab-1)//2)+1, (mid2[0] - (n_lab-1)//2) : (mid2[0] + (n_lab-1)//2)+1].copy()

    # so we don't get squares on our output image
    img_g_out = img_g.copy()
    #draw the rectangles marking the image on the grayscale one
    cv2.rectangle(img_g,((mid1[0] - (n_lab-1)//2) , (mid1[1] - (n_lab-1)//2) ) ,((mid1[0] + (n_lab-1)//2)+1, (mid1[1] + (n_lab-1)//2)+1),(0, 255, 0), 2)
    #cv2.rectangle(img_label,((mid1[0] - (n_lab-1)//2) , (mid1[1] - (n_lab-1)//2) ) ,((mid1[0] + (n_lab-1)//2)+1, (mid1[1] + (n_lab-1)//2)+1),(0, 255, 0), 2)
    cv2.rectangle(img_g,((mid2[0] - (n_lab-1)//2) , (mid2[1] - (n_lab-1)//2) ) ,((mid2[0] + (n_lab-1)//2)+1, (mid2[1] + (n_lab-1)//2)+1),(0, 255, 0), 2)
    #cv2.rectangle(img_label,((mid2[0] - (n_lab-1)//2) , (mid2[1] - (n_lab-1)//2) ) ,((mid2[0] + (n_lab-1)//2)+1, (mid2[1] + (n_lab-1)//2)+1),(0, 255, 0), 2)
    cv2.rectangle(img_g,(min_x,min_y), (max_x, max_y),(0, 255, 0), 2)
    img_g[img_label == 255] = 255
    if display_images:
        cv2.imshow("sub_image_1",img_cropped_1)
        cv2.imshow("sub_label_1",label_cropped_1)
        cv2.imshow("sub_image_2",img_cropped_2)
        cv2.imshow("sub_label_2",label_cropped_2)
        cv2.imshow("img",img_g)
        #cv2.imshow("label",img_label)
        #cv2.imwrite("images\\overlay_two_crop" + str(n) + ".png",img_g)        
        cv2.waitKey(0)

    return (img_cropped_1, label_cropped_1), (img_cropped_2, label_cropped_2), img_g_out

def normalise_image(img):
    """normalise the input image to the range [-1:1]"""
    img_out = (img - 128*np.ones(img.shape))*(1/128.0)
    return img_out
def normalise_label(img):
    """normalise the label to all zeros or ones"""
    img_out = np.zeros(img.shape)
    img_out[img == 255] = 1
    return img_out


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        #bp_sigm = self.bipolar_sigmoid
        self.cnn_1 = nn.Conv2d(1,3,11)
        self.cnn_2 = nn.Conv2d(3,2,11)
        self.cnn_3 = nn.Conv2d(2,1,11)

    def bipolar_sigmoid(self,x):
        return torch.sigmoid(x)*2 - 1
    

    def forward(self, img):
        img = self.cnn_1(img)
        img = self.bipolar_sigmoid(img)
        img = self.cnn_2(img)
        img = self.bipolar_sigmoid(img)
        img = self.cnn_3(img)
        img = self.bipolar_sigmoid(img)
        return img


def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Conv2d') != -1:
        # apply a uniform distribution to the weights and a bias=0
        # -0.3 -> 0.3 comes from the paper
        m.weight.data.uniform_(-0.3, 0.3)
        m.bias.data.fill_(0)

def image_to_tensor(img : np.array):
    """ takes the loaded image slice and exports a tensor. This includes adding a "1" channel as expected by the cnn
    """
    img = torch.tensor(normalise_image(img), dtype = torch.float32)
    return torch.unsqueeze(img,0)
def tensor_to_array(tensor):
    return torch.squeeze(tensor).detach().numpy()

def main():
    filepath = "C:\\Users\\seoir\\git\\replicating_old_papers\\cell_data"

    # Generate the test and train images
    n_1 = 26
    n_2 = 25

    # we take the train and evaluate images from different images to give more general training 
    (tr_1, tr_lb_1), (te_2,te_lb_2), img = extract_two_sub_images(filepath, n_1, False, 97, 67)
    (te_1, te_lb_1), (tr_2,tr_lb_2), _ = extract_two_sub_images(filepath, n_2, False, 97, 67)

    # turn all images to tensors for training
    tr_1 = image_to_tensor(tr_1)
    tr_2 = image_to_tensor(tr_2)
    te_1 = image_to_tensor(te_1)
    te_2 = image_to_tensor(te_2)
    
    tr_lb_1 = image_to_tensor(tr_lb_1)
    tr_lb_2 = image_to_tensor(tr_lb_2)
    te_lb_1 = image_to_tensor(te_lb_1)
    te_lb_2 = image_to_tensor(te_lb_2)

    # Combining the data into batches:
    train_data = torch.stack([tr_1,tr_2])
    train_labels = torch.stack([tr_lb_1,tr_lb_2])

    eval_data = torch.stack([te_1,te_2])
    eval_labels = torch.stack([te_lb_1,te_lb_2])


    # From the paper
    learningRate = 0.05
    momentum = 0.9

    model = Network()
    #this applies the function to each layer of the model
    model.apply(weights_init_uniform)

    loss_function = nn.MSELoss()
    # probably change this later
    optimiser = torch.optim.Adam(model.parameters(), lr=learningRate, betas = (momentum, 0.999))

    #print(model(tr_1))

    iterations = 600
    iter_numbers = []
    losses = []
    losses_eval = []
    iter_numbers_eval = []
    model.train()

    
    for i in tqdm(range(iterations)):
        iter_numbers.append(i)
        optimiser.zero_grad()
        pred = model(train_data)
        loss = loss_function(pred, train_labels)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
        if (i+1)%10 == 0:
            model.eval()
            iter_numbers_eval.append(i)
            losses_eval.append(loss_function(model(eval_data),eval_labels).item())
            model.train()

    plt.figure()
    plt.plot(iter_numbers,losses, label = 'Train')
    plt.plot(iter_numbers_eval,losses_eval, label = 'Evaluation')
    plt.legend()
    plt.xlabel('Iteration Number')
    plt.ylabel('MSE Loss')
    plt.show()
    #im_out = model_tr()
    model.eval()
    #print(tensor_to_array(model(tr_1)))
    cv2.imshow("Train", torch.squeeze(model(tr_1)).detach().numpy())
    cv2.imshow("Eval", torch.squeeze(model(te_1)).detach().numpy())
    cv2.waitKey(0)

    img = image_to_tensor(img)
    cv2.imshow("Full Image", tensor_to_array(model(img)))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()