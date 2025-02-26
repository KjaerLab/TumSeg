import numpy as np
import scipy
import matplotlib
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
import torch

'''
Evaluation functions
'''
def computeRates(Y, P, thress=0.5, cuda=True):
    '''
    Computes the TP, TN, FP and FN
    input:
        Y:          tensor of true labels
        P:          prediction of labels
        thress:     thresshold to be used for prediction
    Output:
        FP, TP, TN and FN
    '''
    label = Y == 1
    pred = P>=thress

    if cuda:
        TP = (label & pred).sum().float()
        TN = (~label & ~pred).sum().float()
        FP = (~label & pred).sum().float()
        FN = (label & ~pred).sum().float()
    else:
        TP = (label & pred).sum()
        TN = (~label & ~pred).sum()
        FP = (~label & pred).sum()
        FN = (label & ~pred).sum()
    
    return TP, TN, FP, FN

def precisionRecallFscore(TP, TN, FP, FN, detach=True):
    '''
    This function computes precision, recall and f1-score
    '''
    if TP + FP == 0:
        precision = TP
    else:
        precision = TP/(TP + FP)
        
    if TP + FN == 0:
        recall = TP
    else:
        recall = TP/(TP + FN)
        
    if precision + recall == 0:
        f1_score = precision
    else:
        f1_score = 2*(precision*recall)/(precision + recall)
    
    if detach:
        return precision.detach().cpu(), recall.detach().cpu(), f1_score.detach().cpu()
    else:
        return precision, recall, f1_score

def getPrecisionRecallFscore(Y, P, thress=0.5, cuda=True, detach=True):
    TP, TN, FP, FN = computeRates(Y,P,thress=thress, cuda=cuda)
    precision, recall, f1_score = precisionRecallFscore(TP, TN, FP, FN, detach=detach)
    return precision, recall, f1_score

def getF1Score(Y, P, thress=0.5, cuda=True, detach=True):
    precision, recall, f1_score = getPrecisionRecallFscore(Y, P, thress=0.5, cuda=cuda, detach=True)
    return f1_score
    


def findSmallROIs(labels, size_thres=0.3):
    '''
    Args:
        labels (array): labels from ndimage.label.
        size_thres (float, optional): size threshold in percent of biggest roi to keep. Defaults to 0.3.

    Returns:
        label_small (list): labels to remove.

    '''
    roi_size = []
    num_features = np.max(labels)
    
    for i in range(1, num_features+1):
        size = (labels == i).sum()
        roi_size.append(size)
        
    roi_size = np.array(roi_size)
    # roi_size = roi_size/roi_size.max()*100
    roi_size = roi_size/roi_size.max()
    
    label_small = np.where(roi_size <= size_thres)[0]+1 # plus 1 to account for offset above
    
    return label_small

def findHighIntensity(labels, CT_in, thres=0.9, max_perc=0.1):
    '''
    Args:
        labels (array): labels from ndimage.label
        CT_in (array): CT image
        thress (float, optional): threshold for what is counted as high intensity. Defaults to 0.9.
        max_perc (float, optional): percentage of high intensity threshold. Defaults to 0.1.

    Returns:
        label_high (list): labels exceeding threshold for being high intensity.
    '''
    label_high = []
    num_features = np.max(labels)
    
    for i in range(1, num_features+1):
        vals = np.array((CT_in[labels == i]))
        q = np.quantile(vals, thres)
        perc = (vals >= q).sum()/len(vals)
        
        if perc >= max_perc:
            label_high.append(i)

    return np.array(label_high)


def postProcessROIs(output, CT_in, classify_thres=0.5, size_thres=0.3, remove_intensity=True, 
                    intensity_thres=0.99, intensity_perc_thres=0.1, verbose=False):
    
    proc_roi = output >= classify_thres
    labels, num_features = scipy.ndimage.label(proc_roi)
    
    if num_features == 0:
        print('WARNING: ROI is empty, returning an empty ROI')
        return proc_roi
    
    if remove_intensity:
        rm1 = findHighIntensity(labels, CT_in, thres=intensity_thres, max_perc=intensity_perc_thres)
    else:
        rm1 = np.array([])
    rm2 = findSmallROIs(labels, size_thres=size_thres)
    
    # collect unique rois to remove 
    rm = list(set(np.hstack((rm1,rm2))))
    
    if verbose:
        print('Found {} ROIs'.format(num_features))
        print('Kept {} ROIs'.format(num_features-len(rm)))
        print('-'*10)
        print('Removed {} intesity ROIs'.format(len(rm1)))
        print('Removed {} size ROIs'.format(len(rm2)))
        print('')
    
    # remove the ROIs labeled for exclusion
    proc_roi[np.isin(labels, rm)] = False
    
    if proc_roi.sum() != 0:
        return proc_roi
    else:
        return output >= classify_thres
    

def locateMice(ct, guess=None, verbose=False, return_fit=False, crop_size=[96, 96], zoom_factor = [1,1,1], debug=False):
    'Locates the mice if there is more than one'
    
    scale_x = ct.shape[1]/960
    scale_y = ct.shape[0]/960
    scale_z = ct.shape[2]/960

    cut_len = int(35*scale_z)
    
    # current crop size
    x_range = int(crop_size[1]/zoom_factor[1])
    y_range = int(crop_size[0]/zoom_factor[0])

    if verbose:
        print('scale x: {:.2f}'.format(scale_x))
        print('scale y: {:.2f}'.format(scale_y))
        print('scale z: {:.2f}'.format(scale_z))
        print('cut length z: {}'.format(cut_len))
        print('')
    
    # clip CT at 
    ct_array_clip = ct.copy()#.astype(np.int16)
    # ct_array_clip[ct_array_clip>4e3] = 4e3
    # ct_array_clip = ct_array_clip[:,:,cut_len:-cut_len]
    
    # max-project in the z-dimension
    ct_proj = ct_array_clip.max(axis=2)
    
    # mean-project along axis x and y axis
    x_mean_proj = ct_proj.mean(axis=0)
    y_mean_proj = ct_proj.mean(axis=1)
    
    # normalize curves for potential plotting
    x_mean_proj = normalize(x_mean_proj, a=0, b=200)
    y_mean_proj = normalize(y_mean_proj, a=0, b=200)

    # smooth out projections
    N = int(50*scale_x)
    x_mean_proj = np.convolve(x_mean_proj, np.ones(N)/N, mode='same')

    N = int(50*scale_y)
    y_mean_proj = np.convolve(y_mean_proj, np.ones(N)/N, mode='same')

    # makes axes
    x_axis = np.arange(len(x_mean_proj))
    y_axis = np.arange(len(y_mean_proj))

    # initial guess
    if guess is None:
        # mu, amp, sigma 
        guess = [300*scale_x, 200*scale_x, 100, 
                 600*scale_x, 200*scale_x, 100]
        
    # fit Gaussians to the x projection
    fit_params_x, fit_x = fitGaussianMix(x_axis, x_mean_proj, guess)
    two_mice_x_axis = 200*scale_x < np.abs(fit_params_x['mu'][0]-fit_params_x['mu'][1])
    if not two_mice_x_axis:
        fit_params_x, fit_x = fitGaussianMix(x_axis, x_mean_proj,  
                                             guess = [np.mean(fit_params_x['mu']),
                                                      np.mean(fit_params_x['amp']),
                                                      np.mean(fit_params_x['sigma'])])
      
    # fit Gaussians two the y projection
    fit_params_y, fit_y = fitGaussianMix(y_axis, y_mean_proj, guess)
    two_mice_y_axis = 200*scale_y < np.abs(fit_params_y['mu'][0]-fit_params_y['mu'][1])
    if not two_mice_y_axis:
        fit_params_y, fit_y = fitGaussianMix(y_axis, y_mean_proj,  
                                             guess = [np.mean(fit_params_y['mu']),
                                                      np.mean(fit_params_y['amp']),
                                                      np.mean(fit_params_y['sigma'])])
    if verbose:
        print('Two mice on x-axis: {}'.format(two_mice_x_axis))
        print('Two mice on y-axis: {}'.format(two_mice_y_axis))
        print('')
        
    num_mice = len(fit_params_x['mu']) * len(fit_params_y['mu'])
        
    mice_slice = {}
    tmp_mice_slice = {}
    
    count = 0
    for x_idx in range(len(fit_params_x['mu'])):
        for y_idx in range(len(fit_params_y['mu'])):    
            tmp_mice_slice[count] = {'x_range': [int(fit_params_x['mu'][x_idx]-x_range//2),
                                                 int(fit_params_x['mu'][x_idx]+x_range//2)],
                                     'y_range': [int(fit_params_y['mu'][y_idx]-y_range//2), 
                                                 int(fit_params_y['mu'][y_idx]+y_range//2)]}
            
            count += 1
        
    sorted_mouse_list = sorted(tmp_mice_slice, key = lambda mouse: tmp_mice_slice[mouse]['x_range'][0], reverse=True)
    
    
    for i in range(num_mice):
        mice_slice['mouse_' + str(i+1)] = tmp_mice_slice[sorted_mouse_list[i]]

    if verbose:
        print('Mice slices: ')
        print(mice_slice)
        print('')
        
    if debug:
        cmap = matplotlib.cm.get_cmap('tab10')
        
        fig, axx = plt.subplots()
        
        axx.imshow(ct_proj, cmap='gray', origin='lower')

        axx.plot(x_mean_proj, alpha=0.7)
        axx.plot(y_mean_proj, np.arange(len(y_mean_proj)), alpha=0.7)

        for x_idx in range(len(fit_params_x['mu'])):
            for y_idx in range(len(fit_params_y['mu'])):
                rect = matplotlib.patches.Rectangle((fit_params_x['mu'][x_idx]-x_range//2,
                                                     fit_params_y['mu'][y_idx]-y_range//2),
                                             x_range, y_range, edgecolor=cmap(i), fill=False)        
                axx.add_patch(rect)
    

        plt.show()

    if return_fit:
        return mice_slice, [fit_x, fit_y]
    else:
        return mice_slice

   
def normalize(x, min_val=None, max_val=None, a=None, b=None):
    '''
    x is the data to be normalized. If min and max is None, the min and max from the data is used. If a and b is None, else 
    the data is normalized between 0 and 1
    '''
    x_ = x.copy()
    
    # 0 to 1 is default
    if all([a is None, b is None]):
        a = 0
        b = 1    
    # check whether a max and min has been given
    if all([min_val is None, max_val is None]):
        x_ = (b-a)*(x_-np.min(x_))/(np.max(x_)-np.min(x_))+a
    else:
        x_ = (b-a)*(x_-min_val)/(max_val-min_val)+a
        
    return x_


def fitGaussianMix(x, y, guess):
    def guassianSum(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            mu = params[i]
            amp = params[i+1]
            sigma = params[i+2]
            y = y + amp * np.exp( -((x - mu)/sigma)**2)
        return y
    
    # fit parameters    
    popt, pcov = curve_fit(guassianSum, x, y, p0=guess)
    # show fit
    fit = guassianSum(x, *popt)
    # pack parameters
    
    fit_params = {'mu': [popt[i] for i in range(0, len(popt), 3)],
                  'amp': [popt[i+1] for i in range(0, len(popt), 3)],
                  'sigma': [popt[i+2] for i in range(0, len(popt), 3)]
        }
    
    return fit_params, fit 

def maybePermute(permute):
    def _maybePermute(subject):
        if permute:
            subject['CT']['data'] = torch.permute(subject['CT']['data'], [0, permute[0]+1, permute[1]+1, permute[2]+1])
        else:
            pass
        return subject
    
    return _maybePermute

