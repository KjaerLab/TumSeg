#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torchio as tio
import argparse
from typing import Optional
import numpy as np
import torch

from tumseg_misc import postProcessROIs, locateMice, maybePermute
from modules import TumSeg, buildSubjectList, runInference, resampleAndPostProcess, saveResults, windowCT


def main(input_path: str, 
         output_path: str,
         device: Optional[str] = None,
         run_uq: bool = False,
         locate_mice: bool = False,
         permute: Optional[str] = None
         ):
    
    
    ''' Setup TumSeg '''
    net_path_A = './networks/network_annotator_A.pt'
    net_path_B = './networks/network_annotator_B.pt'
    net_path_C = './networks/network_annotator_C.pt'
    net_path_all = './networks/network_annotator_A-C.pt'
    
    tumseg = TumSeg(net_path_all, device = args.device)
    # initialize the ensemble 
    tumseg.init_ensemble(net_path_A, net_path_B, net_path_C)
    
    # Attach the post processing module 
    post_proc_kwargs = {
            'classify_thres': 0.5, 
            'size_thres': 0.2, 
            'remove_intensity': False, 
            'verbose': False
            }
    
    post_proc_kwargs_UQ = {
            'classify_thres': 0.5, 
            'size_thres': 0.1, 
            'remove_intensity': True, 
            'verbose': False,
            'intensity_thres': 0.99, 
            'intensity_perc_thres': 0.1
            }
    
    tumseg.attach_post_processor(post_processor = postProcessROIs, 
                                 post_proc_kwargs = post_proc_kwargs)
    
    tumseg.attach_post_processor_UQ(post_processor = postProcessROIs, 
                                 post_proc_kwargs = post_proc_kwargs_UQ)
    
    
    '''Setup data to run '''
    target_pixel_size = (0.420, 0.420, 0.420)
    
    transform = tio.Compose(
        [
         windowCT(-400, 400),
         tio.RescaleIntensity([0,1]),
         tio.Resample(target_pixel_size),
         maybePermute(permute),
         ])
    
    subjects = buildSubjectList(args.input_path)
    print('Found {} scans'.format(len(subjects)))
    
    dataloader = tio.SubjectsDataset(subjects, transform=transform)
    
    
    for subj in dataloader:  
        print('Analyzing: ' + subj['path'])
        if locate_mice:
            mice_slices = locateMice(subj['CT'].numpy()[0], crop_size=[112, 112], debug=False, verbose=False)
        else:
            mice_slices = None
        
        output = runInference(subj, tumseg, mice_slices)
        
        # invert the permute if any
        if permute:
            output = np.transpose(output, np.argsort(permute))

        print('resampling..')
        output = resampleAndPostProcess(output, subj, tumseg, target_pixel_size, locate_mice)
        
        if args.run_uq:
            print('Running Monte-Carlo samples for UQ..')
            predicted_dice = tumseg.runUQ(subj, mice_slices=mice_slices, n_dropouts=100)
            
            # print the predicted Dice score for each 
            for i, uq_pred in enumerate(predicted_dice):
                if uq_pred < 0:
                    uq_pred = 0
                elif uq_pred > 1:
                    uq_pred = 1
                    
                print('')
                print('Expected Dice score for mouse {}: {:.3f}'.format(i+1, uq_pred))
                print('')
            
            saveResults(output, subj, input_path, output_path, uq_pred=predicted_dice)
        else:
            saveResults(output, subj, input_path, output_path, uq_pred=None)

        print('Done.\n\n')
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help="Path to the input must be either a folder containing nifti files or ne a single file ending with .nii or .nii.gz")
    parser.add_argument('-o', '--output_path', type=str, required=False,
                        help="Path to the output directory. If omitted, input_path will be used as output_path.")
    parser.add_argument('--device', type=str, required=False,
                        help="Path to the output directory")
    parser.add_argument('--run-uq', action='store_true', default=False,
                        help="Run UQ (default: False)")
    parser.add_argument('--locate-mice', action='store_true', default=False,
                        help="Run UQ (default: False)")
    parser.add_argument('--permute', nargs='+', type=int,
                        help='List of axes to permute')
    
    args = parser.parse_args()
        
    main(input_path = args.input_path, 
         output_path= args.output_path,
         device = args.device,
         run_uq = args.run_uq,
         locate_mice = args.locate_mice,
         permute = args.permute)















