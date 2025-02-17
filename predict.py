#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torchio as tio
import argparse
from typing import Optional

from tumseg_misc import postProcessROIs
from modules import TumSeg, buildSubjectList, runInference, resampleAndPostProcess, saveResults, windowCT


def main(input_path: str, 
         output_path: str,
         device: Optional[str],
         run_uq: bool = False
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
         tio.Resample(target_pixel_size) 
         ])
    
    subjects = buildSubjectList(args.input_path)
    print('Found {} scans'.format(len(subjects)))
    
    dataloader = tio.SubjectsDataset(subjects, transform=transform)
    
    
    for subj in dataloader:
        print('Analyzing: ' + subj['path'])
        output = runInference(subj, tumseg)
        
        print('resampling..')
        output = resampleAndPostProcess(output, subj, tumseg, target_pixel_size)
        
        if args.run_uq:
            print('Running Monte-Carlo samples for UQ..')
            uq_pred = tumseg.runUQ(subj)
            if uq_pred < 0:
                uq_pred = 0
            elif uq_pred > 1:
                uq_pred = 1
                
            print('')
            print('Expected Dice score: {:.3f}'.format(uq_pred))
            print('')
            
            saveResults(output, subj, args.input_path, args.output_path, uq_pred=uq_pred)
        else:
            saveResults(output, subj, args.input_path, args.output_path, uq_pred=None)

        print('Done.\n\n')
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help="Path to the input must be either a folder containing nifti files or ne a single file ending with .nii or .nii.gz")
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help="Path to the output directory")
    parser.add_argument('--device', type=str, required=False,
                        help="Path to the output directory")
    parser.add_argument('--run_uq', action='store_true', default=False,
                        help="Run UQ (default: False)")
    
    args = parser.parse_args()
        
    main(input_path = args.input_path, 
        output_path= args.input_path,
        device = args.input_path,
        run_uq = args.input_path)















