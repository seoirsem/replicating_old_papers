import cv2
import os
import numpy as np
from torch import nn
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
from os.path import exists

def extract_two_sub_images(filepath : str, image_number : int, display_images : bool, n_image : int, n_label : int):
    """extracts two subimages from the given image number. Returns the label and the image

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
    #print(label)
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

    # one of the images falls off the edge
    if min_y <15:
        min_y = 15
    return (img_cropped_1, label_cropped_1), (img_cropped_2, label_cropped_2), img_g_out, \
            img_g_out[(min_y-15):(max_y+15), (min_x-15):(max_x+15)], img_label, img_label[(min_y-15):(max_y+15), (min_x-15):(max_x+15)]


def normalise_image(img):
    """normalise the input image to the range [-1:1]"""
    img_out = (img - 128*np.ones(img.shape))*(1/128.0)
    return img_out

def normalise_positive(img):
    """Normalise the input image to the range [0:1]"""
    img_out = (img - 255*np.ones(img.shape))*(1/255.0)
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
        intermediary = []
        img = self.cnn_1(img)
        img = self.bipolar_sigmoid(img)
        intermediary.append(img)
        img = self.cnn_2(img)
        img = self.bipolar_sigmoid(img)
        intermediary.append(img)
        img = self.cnn_3(img)
        img = self.bipolar_sigmoid(img)
        return img, intermediary

class Network_ReLU(nn.Module):

    def __init__(self):
        super(Network_ReLU, self).__init__()
        #bp_sigm = self.bipolar_sigmoid
        self.cnn_1 = nn.Conv2d(1,3,11)
        self.cnn_2 = nn.Conv2d(3,2,11)
        self.cnn_3 = nn.Conv2d(2,1,11)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        img = self.cnn_1(img)
        img = self.relu(img)
        img = self.cnn_2(img)
        img = self.relu(img)
        img = self.cnn_3(img)
        img = self.relu(img)
        #img = self.sigmoid(img)
        return img


def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Conv2d') != -1:
        # apply a uniform distribution to the weights and a bias=0
        # -0.3 -> 0.3 comes from the paper
        m.weight.data.uniform_(-0.3, 0.3)
        m.bias.data.fill_(0)

def image_to_tensor(img : np.array, normalise_pos : bool):
    """ takes the loaded image slice and exports a tensor. This includes adding a "1" channel as expected by the cnn
    """
    if normalise_pos:
        img = torch.tensor(normalise_positive(img), dtype = torch.float32)
    else:
        img = torch.tensor(normalise_image(img), dtype = torch.float32)
    return torch.unsqueeze(img,0)

def tensor_to_array(tensor):
    return torch.squeeze(tensor).detach().numpy()

def post_process(img : np.array) -> np.array:

    n = 190
    img[img<n] = 0
    img[img>=n] = 255
    k = 3
    k1 = np.ones((k,k),np.uint8)
    k2 = np.ones((k+1,k+1),np.uint8)
    #img = cv2.erode(img,k1,iterations = 1)
    #img = cv2.dilate(img,k2,iterations = 1)
    img = ridge_finding(img)
    return img

def ridge_finding(img):
    (a, b) = img.shape
    im2 = np.zeros(img.shape)
    for i in range(1,a-1):
        for j in range(1,b-1):
            if (img[i,j] > img[i-1,j]) and (img[i,j] > img[i+1,j]):
                im2[i,j] = 255
            elif (img[i,j] > img[i,j-1]) and (img[i,j] > img[i,j+1]):
                im2[i,j] = 255
            elif (img[i,j] > img[i-1,j-1]) and (img[i,j] > img[i+1,j+1]):
                im2[i,j] = 255
            elif (img[i,j] > img[i-1,j+1]) and (img[i,j] > img[i+1,j-1]):
                im2[i,j] = 255
    return im2

def train_paper(filepath):
        n_1 = 26
        n_2 = 25
        improved = False
        model, iter_numbers, losses, iter_numbers_eval, losses_eval, img = train_like_paper(filepath, n_1, n_2, improved)

        plt.figure()
        plt.plot(iter_numbers,losses, label = 'Train')
        plt.plot(iter_numbers_eval,losses_eval, label = 'Evaluation')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Iteration Number')
        plt.ylabel('MSE Loss')
        plt.show()
        #im_out = model_tr()
        model.eval()
        #print(tensor_to_array(model(tr_1)))
        #cv2.imshow("Train", torch.squeeze(model(tr_1)).detach().numpy())
        #cv2.imshow("Eval", torch.squeeze(model(te_1)).detach().numpy())
        #cv2.waitKey(0)

        img = image_to_tensor(img, improved)
        img = tensor_to_array(model(img))
        print(img)
        cv2.imshow("Full Image", img)
        cv2.imwrite("image_for_process.png", img*128 + np.ones(img.shape)*128)
        cv2.waitKey(0)
        return model


def train_like_paper(filepath : str, n_1 : int, n_2 : int, improved : bool):
    """
    Trains the network exactly as the paper does
    """

        # we take the train and evaluate images from different images to give more general training 
    (tr_1, tr_lb_1), (te_2,te_lb_2), img, _, _, _ = extract_two_sub_images(filepath, n_1, False, 97, 67)
    (te_1, te_lb_1), (tr_2,tr_lb_2), _, _, _, _ = extract_two_sub_images(filepath, n_2, False, 97, 67)

    # turn all images to tensors for training
    tr_1 = image_to_tensor(tr_1, improved)
    tr_2 = image_to_tensor(tr_2, improved)
    te_1 = image_to_tensor(te_1, improved)
    te_2 = image_to_tensor(te_2, improved)
    
    tr_lb_1 = image_to_tensor(tr_lb_1, improved)
    tr_lb_2 = image_to_tensor(tr_lb_2, improved)
    te_lb_1 = image_to_tensor(te_lb_1, improved)
    te_lb_2 = image_to_tensor(te_lb_2, improved)

    # Combining the data into batches:
    train_data = torch.stack([tr_1,tr_2])
    train_labels = torch.stack([tr_lb_1,tr_lb_2])

    eval_data = torch.stack([te_1,te_2])
    eval_labels = torch.stack([te_lb_1,te_lb_2])


    # From the paper
    learningRate = 0.05
    momentum = 0.9

    if improved:
        model = Network_ReLU()
    else:
        model = Network()
    #this applies the function to each layer of the model
    if not improved:
        model.apply(weights_init_uniform)

    loss_function = nn.MSELoss()
    # probably change this later
    optimiser = torch.optim.Adam(model.parameters(), lr=learningRate, betas = (momentum, 0.999))

    iterations = 400
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

    return model, iter_numbers, losses, iter_numbers_eval, losses_eval, img

#def train_large_sample()

def train_model(optimiser, loss_function, train_data, train_labels, iterations, model, eval_data, eval_labels, losses, iter_numbers):
    """Trains the given model for the given number of iterations (epochs). Records eval loss every epoch"""
    losses_eval = []
    iter_numbers_eval = []
    model.train()    
    for i in tqdm(range(iterations)):
        for j in range(train_data.shape[0]):
            d = train_data[j,:,:,:]
            #d = torch.unsqueeze(d,0)
            l = train_labels[j,:,:,:]
            optimiser.zero_grad()
            pred = model(d)
            #print(pred[0,0,:])
            loss = loss_function(pred, l)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            iter_numbers.append(iter_numbers[-1]+1)

        model.eval()
        iter_numbers_eval.append(i*train_data.shape[0] + j)
        losses_eval.append(loss_function(model(eval_data),eval_labels).item())
        model.train()

    return model, iter_numbers, losses, iter_numbers_eval, losses_eval


def save_data_arrays_to_file(filepath : str):
    """Saves image arrays to file from 1->28. These can be more easily manipulated later"""
    shapes = np.zeros((28, 2))
    images = []
    labels = []
    for i in range(1,29):
        (_, _), (_, _), _, img, test, label = extract_two_sub_images(filepath, i, False, 97, 67)
        images.append(img)
        labels.append(label)
        shapes[i-1,:] = img.shape
    # now we take the shape as the smallest shape which covers all examples
    # a bit wasteful of data but makes training cleaner
    batch_shape = (int(min(shapes[:,0])), int(min(shapes[:,1])))
    batch_images = np.zeros((28, *batch_shape))        
    batch_labels = np.zeros((28, *batch_shape))        
    for i in range(28):
        img = images[i]
        shape = img.shape
        delta = (shape[0] - batch_shape[0], shape[1] - batch_shape[1])
        img = img[math.ceil(delta[0]/2):shape[0] - math.floor(delta[0]/2), math.ceil(delta[1]/2): shape[1] - math.floor(delta[1]/2)]
        #cv2.imshow("A",img)
        #cv2.waitKey(0)
        label = labels[i][math.ceil(delta[0]/2):shape[0] - math.floor(delta[0]/2), math.ceil(delta[1]/2): shape[1] - math.floor(delta[1]/2)]
        batch_images[i,:,:] = img
        batch_labels[i,:,:] = label
    
    #img = batch_images[1,:,:]/255
    print(batch_images.shape)
    np.save("image_data.npy",batch_images)
    np.save("label_data.npy",batch_labels)
    print("Files saved!")


def train_custom(save_model : bool, load_model : bool, plot_figs : bool, filename : str):
    """I train my network slightly modified from the paper"""
    # load data and labels
    batch_images = np.load("image_data.npy")
    batch_labels = np.load("label_data.npy")

    batch_labels[batch_labels<200] = 0         
    batch_labels[batch_labels>=200] = 1         
    # normalise data
    batch_images = batch_images/255
    train_data = torch.tensor(batch_images[:-2,:,:], dtype = torch.float32)
    eval_data = torch.tensor(batch_images[-2:,:,:], dtype = torch.float32)
    train_labels = torch.tensor(batch_labels[:-2,15:-15,15:-15], dtype = torch.float32)
    eval_labels = torch.tensor(batch_labels[-2:,15:-15,15:-15], dtype = torch.float32)

    train_data = torch.unsqueeze(train_data,1)
    train_labels = torch.unsqueeze(train_labels,1)
    eval_data = torch.unsqueeze(eval_data,1)
    eval_labels = torch.unsqueeze(eval_labels,1)

#        print(train_labels)
    # model
    learningRate = 0.005
    momentum = 0.9
    model = Network()
    optimiser = torch.optim.Adam(model.parameters(), lr=learningRate, betas = (momentum, 0.999))
    iter_numbers = [0]
    losses = [0]

    if load_model and exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        iter_numbers = checkpoint['iter_numbers']
        #initialEpoch = checkpoint['epoch']
        losses = checkpoint['losses']
        print('Loaded model at ' + filename)

    loss_function = nn.MSELoss()
    iterations = 30
    for i in range(iterations):
        model, iter_numbers, losses, iter_numbers_eval, losses_eval = train_model(optimiser, loss_function, train_data, train_labels, 1, model, eval_data, eval_labels, losses, iter_numbers)
        eval = model(eval_data)
        eval = tensor_to_array(eval)
        img = eval[0,:,:]
        cv2.imwrite("training_epoch_"+str(i)+".png",img*255)
        print("epoch[{}/{}], with a final loss of {:.3f}".format(i+1,iterations,losses[-1]))

    if plot_figs:
        plt.figure()
        plt.plot(iter_numbers,losses, label = 'Train')
        plt.plot(iter_numbers_eval,losses_eval, label = 'Evaluation')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Iteration Number')
        plt.ylabel('MSE Loss')
        plt.show()
        eval = model(eval_data)
        eval = tensor_to_array(eval)
        img = eval[0,:,:]
        cv2.imshow("image",img)
        cv2.waitKey(0)

    filename = "model.pt"
    if save_model:
        torch.save({
        #'epoch': epochs[-1],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'losses': losses,
        'iter_numbers' : iter_numbers
        }, filename)
        print('Model saved as "' + filename + '"')



def main():
    postprocess = False
    if postprocess:
        """applies some thresholding etc from the paper"""
        img = cv2.imread("image_for_process.png", cv2.IMREAD_GRAYSCALE)
        #print(img)
        cv2.imshow("image",img)
        img2 = post_process(img)
        cv2.imshow("processed", img2)
        cv2.waitKey(0)
     
    filepath = "C:\\Users\\seoir\\git\\replicating_old_papers\\cell_data"
    

    more_data = True
    load_model = True
    filename = "model.pt"
    filename_paper = "model_paper.pt"
    save_model = False
    plot_figs = True
    train_model = False


    if train_model:
        train_custom(save_model, load_model, plot_figs, filename)
        model_paper = train_paper(filepath)
        if save_model:
            torch.save({
            #'epoch': epochs[-1],
            'model_state_dict': model_paper.state_dict(),
            }, filename_paper)
            print('Model saved as "' + filename_paper + '"')
    if load_model and exists(filename):
        model = Network()
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Loaded model at ' + filename)
        model.eval()

        model_paper = Network()
        checkpoint = torch.load(filename_paper)
        model_paper.load_state_dict(checkpoint['model_state_dict'])
        print('Loaded model at ' + filename_paper)

    i_image = 1
    (img, _), (_, _), img2, _, _, _ = extract_two_sub_images(filepath, i_image, False, 97, 67)
    pred_larger_network, inter_new = model(image_to_tensor(img,False))
    pred_larger_network = tensor_to_array(pred_larger_network)
    pred_smaller_network, inter_paper = model_paper(image_to_tensor(img,False))
    pred_smaller_network = tensor_to_array(pred_smaller_network)

    inter_new_1 = tensor_to_array(inter_new[0][0,:,:])
    inter_new_2 = tensor_to_array(inter_new[1][0,:,:])
    inter_paper_1 = tensor_to_array(inter_paper[0][0,:,:])
    inter_paper_2 = tensor_to_array(inter_paper[1][0,:,:])

    #cv2.imshow("New Network 1",inter_new_1)
    #cv2.imshow("New Network 2",inter_new_2)
    cv2.imshow("New Network out",pred_larger_network)
    #cv2.imshow("Old Network 1", inter_paper_1)
    #cv2.imshow("Old Network 2", inter_paper_2)
    cv2.imshow("Old Network out", pred_smaller_network)
    cv2.imshow("Image", img)
    cv2.waitKey(0)

    #cv2.imwrite("images/network_comparisons/pred_smaller_"+str(i_image)+".png",pred_smaller_network*255)
    #cv2.imwrite("images/network_comparisons/pred_larger_"+str(i_image)+".png",pred_larger_network*255)
    #cv2.imwrite("images/network_comparisons/image_"+str(i_image)+".png",img)

    save_layer_data(pred_larger_network,inter_new,i_image,"new_network")
    save_layer_data(pred_smaller_network,inter_paper,i_image,"old_network")
    folder = "images/network_layers/"
    cv2.imwrite(folder + "image_" + str(i_image) + "_output.png",img)

def save_layer_data(pred_out, layer_data, number : int, name : str):
    folder = "images/network_layers/"
    cv2.imwrite(folder + name + "_" + str(number) + "_output.png",pred_out*255)
    for i, layer in enumerate(layer_data):
        for j in range(layer.shape[0]):
            img = tensor_to_array(layer[j,:,:])
            cv2.imwrite(folder + name + "_" + str(number) + "_layer_" + str(i) + "_channel_" + str(j) + ".png",img*255)




if __name__ == "__main__":
    main()