# CrowdAnomalyDetection
A program for analyzing and grading crowd-simulations against real data.

## Usage
Training data is included in students001.vsp and students003.vsp  
Copied from:  
[https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data]

Test data needs to follow the format described in crowd_file_format.txt

Run the program from the comandline with:  
$python3 dyads.py TRAINING-DATA TEST-DATA

Results are shown as SCORE and MISSING and are the avaraged depth of coresponding calculations, lower is better.

## Based on
*A Data-Driven Framework for Visual Crowd Analysis*  
Charalambous, Panayiotis, et al.

*Multi-criteria anomaly detection using Pareto Depth Analysis.*  
Hsiao, Ko-Jen, et al. 
