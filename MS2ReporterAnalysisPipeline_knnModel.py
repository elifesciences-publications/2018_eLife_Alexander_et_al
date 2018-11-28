# -*- coding: utf-8 -*-
get_ipython().magic('matplotlib')

"""
    Created on Wed Aug 24 20:38:36 2016
    Modified on 1/6/2018 to include goodness of fit filter.  After gaussian fit on local image data, a goodness of fit calculation is performed on a 8x8 pixel grid centered on the gaussian peak.  Poor gaussian fits (R^2 < 0.5) are filtered out.  I've noticed that this allows for robust calling of MS2 spots identified by eye without too many false positive results.  This version of pipeline with only highlight MS2 signal that has passed filter analysis.
    @author: Jeff
"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import argparse
import pickle
from skimage import io
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import pyMods as pyM  ###User-defined module
import gaussianfunctions as gf


####Get Options Data #####
scalebar_invoked = '-b' in sys.argv or '--scalebar' in sys.argv
stamp_invoked = '-s' in sys.argv or '--stamp' in sys.argv

parser = argparse.ArgumentParser(description='This is the analysis pipeline for quantitate MS2 time-lapse max Z projection datasets.  Requires xyt dataset (single channel)')
parser.add_argument('-i',
                    dest="image",
                    required=True,
                    help='REQUIRED:  Directory containing images to quantitate.  Images should be only one channel.',
                    action='store')

parser.add_argument('-d', '--data',
                    required=True,
                    help='REQUIRED: Table that contains tracking data.  Should be tab-delimited for columns Image\tX Value (pixel)\tY Value (pixel)\tZ Value (pixel)\tT Value (frame)')

parser.add_argument('-o', '--model',
                    required=True,
                    help='REQUIRED: Path to k-Nearest Neighbor model object')

parser.add_argument('-n', '--noise',
                    action='store_true',
                    default=False,
                    help='OPTIONAL: Include a noise filter, which removes detected MS2 signal that are separated by other detected gaussians by >3 frames (i.e. removes high frequency MS2 bursts)')

parser.add_argument('-s', '--stamp',
                    action='store_true',
                    default=False,
                    help='OPTIONAL: Include time stamp on movie.')

parser.add_argument('-m', '--marker',
                    action='store_true',
                    default=False,
                    help='OPTIONAL: Annotate each detect Gaussian with red circle')

parser.add_argument('-a', '--animate',
                    action='store_true',
                    default=False,
                    help='OPTIONAL: Output annotated movies with MS2 signal detection.')

parser.add_argument('-b', '--scalebar',
                    type=float,
                    default=0,
                    help='OPTIONAL: Annotate image with scale bar of user defined length (in um)')

parser.add_argument('-p', '--pixel',
                    type=float,
                    default=0,
                    required=scalebar_invoked,
                    help='REQUIRED IF -b IS INVOKED: Pixel size of image(s) in (um)')

parser.add_argument('-t', '--time',
                    type=int,
                    default=20,
                    required=stamp_invoked,
                    help='REQUIRED IF -s IS INVOKED: Time interval between frames. Default value: 20s')

parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help='OPTIONAL: Verbose Mode')

args = parser.parse_args()

### Initialize Variables ####
image_dir = args.image
localization_table = args.data
model_path = args.model
ratio = 1.1 ###Hardwired in this version
sigma = 4.0 ###Hardwired in this version
stamp = args.stamp
verbose = args.verbose
scalebar = args.scalebar
time_interval = args.time
use_marker = args.marker
pixel_size = args.pixel
animate = args.animate
noise_filter = args.noise

####Load K-Nearest Neighbor Model
with open(model_path, 'rb') as f:
    knn = pickle.load(f)

os.chdir(image_dir);
image_list = glob.glob('*.tif');

df = pd.read_table(localization_table);
df.columns = ['Image', 'X Value (pixel)', 'Y Value (pixel)', 'Z Value (slice)', 'T Value (frame)']
outdf = df.copy()
outdf = outdf.assign(Gaussian_Height = 0, Gaussian_Volume = 0, Background = 0, X_Location = 0, Y_Location = 0, X_Sigma = 0, Y_Sigma = 0, Local_Median = 0, Norm_Height = 0, Norm_Height_Guess = 0, Norm_Height_Filter = 0, R_Squared = 0, Gaussian_Height_Threshold=ratio, Gaussian_Width_Threshold=sigma, Quality_Score = 0, Gauss_Filter=False, Noise_Filter=False, Pass_Filter=False);
pi = 3.14159;
local_size = 20;
gauss_halfwidth = 6;
local_halfwidth = local_size / 2;
local_offset = int(gauss_halfwidth / 2);
pass_filter = True;
frame_number = 0

if noise_filter:
    output_dir = str(ratio) + "_min_height_" + str(sigma) + "_sigma_noise_filterON/"
else:
    output_dir = str(ratio) + "_min_height_" + str(sigma) + "_sigma_noise_filterOFF/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir);
else:
    print('Output directory ' + output_dir + ' already exists.');

os.chdir(output_dir);


#########
## Functions
#########

#define model function and pass independant variables x and y as a list
def getBounds(x_location, y_location, local_halfwidth, local_size, image_spec):
    if (x_location - local_halfwidth) < 0:
        local_x_min = 0;
        local_x_max = int(local_size)
    elif (x_location + local_halfwidth) > image_spec.x_size:
        local_x_min = int(image_spec.x_size - local_size)
        local_x_max = image_spec.x_size
    else:
        local_x_min = int(x_location - local_halfwidth);
        local_x_max = int(x_location + local_halfwidth);
    
    if (y_location - local_halfwidth) < 0:
        local_y_min = 0;
        local_y_max = int(local_size)
    elif (y_location + local_halfwidth) > image_spec.y_size:
        local_y_min = int(image_spec.y_size - local_size)
        local_y_max = image_spec.y_size
    else:
        local_y_min = int(y_location - local_halfwidth);
        local_y_max = int(y_location + local_halfwidth);
    
    return (local_x_min, local_x_max, local_y_min, local_y_max)


                                 
def update (frame_number):
    gauss_x = np.zeros(num_loci)
    gauss_y = np.zeros(num_loci)
    locus_y = np.zeros(num_loci)
    locus_x = np.zeros(num_loci)
    gauss_signal = np.zeros(num_loci)
    
    data = im[:,:, frame_number];
    implt.set_data(data);

    if stamp:
        mins, sec = divmod((frame_number * time_interval), 60)
        frame_string = str(mins) + ":00"; #This version skips annotating every frame but looks nicer in movies.
        frame_text.set_text(frame_string)
    for n in range(0, num_loci):
        matched_row = (outdf['Image'] == uloci[n]) & (outdf['T Value (frame)'] == (frame_number + 1))
        locus_y[n] = outdf.loc[matched_row,'Y Value (pixel)'].values[0]
        locus_x[n] = outdf.loc[matched_row,'X Value (pixel)'].values[0]
        gauss_x[n] = outdf.loc[matched_row,'X_Location'].values[0]
        gauss_y[n] = outdf.loc[matched_row,'Y_Location'].values[0]
        if (outdf.loc[matched_row,'Pass_Filter'].values):
            gauss_signal[n] = outdf.loc[matched_row,'Norm_Height_Filter'].values[0]

        ########
        ## Gaussian Fit
        ########
        
        (local_x_min, local_x_max, local_y_min, local_y_max) = getBounds(locus_x[n], locus_y[n], local_halfwidth, local_size, image_spec)
        roi[n].set_bounds(local_x_min, local_y_min, local_size, local_size);
        labels[n].set_position(np.stack(((local_x_min + 3), (local_y_min - 3))))
        
    if use_marker:
        scat_points = np.stack((gauss_x, gauss_y), axis=-1)
        scat.set_offsets(scat_points)
        plt.setp(scat, linewidth=gauss_signal * 0.5)


###########
###########
###########

with PdfPages('MS2_expression_plots.pdf') as pdf:
    print('Performing MS2 transcriptional signal detection.....')
    for i in image_list:
        cur_image = i;
        if verbose:
            print(cur_image)
        im = io.imread(image_dir + cur_image);
        im = pyM.orderImageDim(im, 1, 'xyt')
        image_spec = pyM.getImageShape(im, 1, 'xyt')
        m = re.search('(Batch\d+_\d+_XY\d+).*.tif', cur_image);
        match = m.group(1);
        loci = df[df.Image.str.contains(match)];

        if not loci.empty:
            uloci = loci.Image.unique();
            num_loci = uloci.size;
            for l in uloci:
                if verbose:
                    print('Working on item: ' + l)
                local_data = np.zeros((local_size, local_size, image_spec.frames), np.uint16);
                data_frames = []
                for count in range(1, image_spec.frames + 1):
                    if (not df.loc[(df['Image'] == l) & (df['T Value (frame)'] == count),:].empty):
                        data_frames.append(count)
                for f in range(1, image_spec.frames + 1):
                    data = im[:,:, (f-1)];
                    
                    ##Get tracking data for gaussian search
                    locus_loc_y = 0;
                    locus_loc_x = 0;
                    local_x_pixels = local_size;
                    local_y_pixels = local_size;
                    x_local = np.linspace(0, local_x_pixels - 1, local_x_pixels);
                    y_local = np.linspace(0, local_y_pixels - 1, local_y_pixels);
                    x_local,y_local = np.meshgrid(x_local, y_local);
                    if (df.loc[(df['Image'] == l) & (df['T Value (frame)'] == (f)),:].empty): ##If time point has no tracking data associated, use the nearest time point data
                        use_frame = min(data_frames, key=lambda x:abs(x-f))
                        locus_loc_y = df.loc[(df['Image'] == l) & (df['T Value (frame)'] == use_frame),'Y Value (pixel)'].values[0];
                        locus_loc_x = df.loc[(df['Image'] == l) & (df['T Value (frame)'] == use_frame),'X Value (pixel)'].values[0];
                        outdf = outdf.append(pd.DataFrame([[l, ratio, sigma, locus_loc_x, locus_loc_y, (f), 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, np.nan, False, False, False]],
                                                          columns=('Image', 'Gaussian_Height_Threshold', 'Gaussian_Width_Threshold', 'X Value (pixel)', 'Y Value (pixel)', 'T Value (frame)', 'Gaussian_Height', 'Gaussian_Volume', 'Background', 'X_Location', 'Y_Location', 'X_Sigma', 'Y_Sigma', 'Local_Median', 'Norm_Height', 'Norm_Height_Guess', 'Norm_Height_Filter', 'R_Squared', 'Quality_Score', 'Gauss_Filter', 'Noise_Filter', 'Pass_Filter')));
                    else:
                        locus_loc_y = df.loc[(df['Image'] == l) & (df['T Value (frame)'] == f),'Y Value (pixel)'].values[0];
                        locus_loc_x = df.loc[(df['Image'] == l) & (df['T Value (frame)'] == f),'X Value (pixel)'].values[0];
                
                
                    ########
                    ## Gaussian Fitting
                    ########
                    
                    ## Get ROI coordinates
                    (local_x_min, local_x_max, local_y_min, local_y_max) = getBounds(locus_loc_x, locus_loc_y, local_halfwidth, local_size, image_spec)
                    ## Extract ROI data
                    local_data[:,:, (f-1)] = data[local_y_min:local_y_max, local_x_min:local_x_max]; ##0-based numbering
                    cur_frame_local_data = local_data[:,:, (f-1)]; ##0-based numbering
                    local_max_value = cur_frame_local_data.max();

                    ## Get Gaussian parameter guesses
                    y_guess, x_guess = np.unravel_index(cur_frame_local_data[5:15, 5:15].argmax(), cur_frame_local_data[5:15,5:15].shape); ##Guess location of gaussian to be location of brightest pixel
                    y_guess = y_guess + 5
                    x_guess = x_guess + 5
                    off_guess = np.median(cur_frame_local_data); ##Guess offset of gaussian to be median pixel value
                    local_median = off_guess
                    height_low_bound = (ratio * off_guess) - off_guess; ##Set lower bound of gaussian height to be 'ratio' fold above background
                    amp_guess = local_max_value - off_guess; ##Guess height of gaussian to be max pixel value minus median pixel value
                    
                    local_spec = pyM.getImageShape(cur_frame_local_data, 1, 'xy')
                    ## Extract image data around gaussian guess for fitting
                    (gauss_x_min, gauss_x_max, gauss_y_min, gauss_y_max) = getBounds(x_guess, y_guess, 5, 10, local_spec)
                    gauss_data = cur_frame_local_data[gauss_y_min:gauss_y_max, gauss_x_min:gauss_x_max]
                    gauss_y_guess, gauss_x_guess = np.unravel_index(gauss_data.argmax(), gauss_data.shape) ##Get gaussian location guess with refined coordinates
                    initial_guess = (amp_guess, gauss_x_guess, gauss_y_guess, 1.0, 1.0, off_guess)
                    bound_tup = ([height_low_bound, 2, 2, 0, 0, 0], [+np.Inf, 8, 8, sigma, sigma, +np.Inf])
                    x_local_gauss = np.linspace(0, 10 - 1, 10)
                    y_local_gauss = np.linspace(0, 10 - 1, 10)
                    x_local_gauss,y_local_gauss = np.meshgrid(x_local_gauss, y_local_gauss)
                    param = gf.tryGaussian(gf.twoD_Gaussian, (x_local_gauss,y_local_gauss), gauss_data, initial_guess, bound_tup) ##Fit gaussian
                    if (param.amp != 0):  ##If good fit
                        Rsqr = gf.getRSquared(gf.twoD_Gaussian, gauss_data, param, 10)
                    else:
                        Rsqr = 0
                    ## Record data in table
                    matched_row = (outdf['Image'] == l) & (outdf['T Value (frame)'] == f)
                    outdf.loc[matched_row,'Gaussian_Height'] = param.amp
                    outdf.loc[matched_row,'Gaussian_Volume'] = param.signal
                    outdf.loc[matched_row,'Background'] = param.offset
                    outdf.loc[matched_row,'X_Location'] = param.center_x + (local_x_min + gauss_x_min)
                    outdf.loc[matched_row,'Y_Location'] = param.center_y + (local_y_min + gauss_y_min)
                    outdf.loc[matched_row,'X_Sigma'] = param.sigma_x
                    outdf.loc[matched_row,'Y_Sigma'] = param.sigma_y
                    outdf.loc[matched_row,'Local_Median'] = local_median
                    outdf.loc[matched_row,'Norm_Height_Guess'] = param.norm_height_guess
                    outdf.loc[matched_row,'R_Squared'] = Rsqr
        
                ## Normalize Gaussian height based on local background (value > 1).  If no gaussian is fit, use local_median as the signal (for approx value of 1)
                normalization_factor = np.median(local_data)
                for f in range(1, image_spec.frames + 1):
                    matched_row = (outdf['Image'] == l) & (outdf['T Value (frame)'] == f)
                    if not np.isnan(outdf.loc[matched_row,'X_Sigma'].values):
                        outdf.loc[matched_row,'Norm_Height'] = (outdf.loc[matched_row,'Gaussian_Height'] + outdf.loc[matched_row,'Background']) / normalization_factor
                    else:
                        outdf.loc[matched_row,'Norm_Height'] = outdf.loc[matched_row,'Local_Median'] / normalization_factor
                        outdf.loc[matched_row,'Quality_Score'] = 0
            
            
                ### Burst Classification using K-Nearest Neighbor Model
                knn_data = outdf.loc[(outdf['Image'] == l),:]
                knn_data = knn_data.sort_values(by=['T Value (frame)'])
                knn_data = knn_data.reset_index(drop=True)
                nrow = len(knn_data.index)
                params = np.zeros((nrow, 4))
                for j in range(0, nrow):
                    params[j, 0] = knn_data['Norm_Height'].iloc[j]
                    params[j, 1] = knn_data['X_Sigma'].iloc[j]
                    params[j, 2] = knn_data['Y_Sigma'].iloc[j]
                    params[j, 3] = knn_data['R_Squared'].iloc[j]
                

                MS2_knn = knn.predict(params)
                for k in range(0, nrow):
                    if MS2_knn[k] == 0:
                        outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (k + 1)), 'Gauss_Filter'] = False
                    elif MS2_knn[k] == 1:
                        outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (k + 1)), 'Gauss_Filter'] = True
                    else:
                        print("Error.  This value should not be generated")

                for f in range(1, image_spec.frames + 1):
                    ##Filter out noise (Gaussian fits that last for only one frame)
                    noise = True
                    if (outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)), 'Gauss_Filter'].values == True):
                        for loop in [-3, -2, -1, 1, 2, 3]:
                            if (outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f + loop)), 'Gauss_Filter'].values == True):
                                noise = False
                        if noise == True:
                            outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)), 'Noise_Filter'] = False
                            outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)),'Norm_Height_Filter'] = outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)),'Local_Median'] / normalization_factor;
                        else:
                            outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)), 'Noise_Filter'] = True
                            outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)),'Norm_Height_Filter'] = (outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)),'Gaussian_Height'] + outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)),'Background']) / normalization_factor;

                    ###Set Pass Filter
                    if (noise_filter):
                        outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)), 'Pass_Filter'] = outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)), 'Noise_Filter']
                    else:
                        outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)), 'Pass_Filter'] = outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)), 'Gauss_Filter']
                    ###Set filtered MS2 signal
                    if (outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)), 'Pass_Filter'].values == True):
                        outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)),'Norm_Height_Filter'] = (outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)),'Gaussian_Height'] + outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)),'Background']) / normalization_factor;
                    else:
                        outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)),'Norm_Height_Filter'] = outdf.loc[(outdf['Image'] == l) & (outdf['T Value (frame)'] == (f)),'Local_Median'] / normalization_factor;

###Print out graphs
    graph_location = 0
    ax = [None] * 6
    graph = plt.figure()
    gs  = gridspec.GridSpec(2, 3)
    for i in image_list:
        cur_image = i;
        m = re.search('(Batch\d+_\d+_XY\d+).*.tif', cur_image);
        match = m.group(1);
        loci = df[df.Image.str.contains(match)];
        if not loci.empty:
            uloci = loci.Image.unique();
            for l in uloci:
                outdf.sort_values(['Image', 'T Value (frame)'], inplace=True);
                ax[graph_location] = graph.add_subplot(gs[graph_location])
                plt.title('Gaussian Estimate of MS2cp Signal\n' + l)
                plt.xlabel('Time (frames)')
                plt.ylabel('MS2cp Signal\n(Height off Fit Gaussian (a.u.)')
                plt.ylim(0, 3)
                for item in ([ax[graph_location].title, ax[graph_location].xaxis.label, ax[graph_location].yaxis.label] +
                             ax[graph_location].get_xticklabels() + ax[graph_location].get_yticklabels()):
                                item.set_fontsize(7)
                ax[graph_location].plot(outdf.loc[outdf.Image == l, ['T Value (frame)']], outdf.loc[outdf.Image == l, ['Norm_Height_Filter']], color='blue', linestyle='-', label='Low-pass Filter')
                legend = ax[graph_location].legend(loc='upper right', fontsize=5)

                graph_location = graph_location + 1
                if (graph_location >= 6):
                    gs.tight_layout(graph)
                    pdf.savefig(graph)
                    plt.close('graph')
                    graph_location = 0
                    graph = plt.figure()
                    gs  = gridspec.GridSpec(2, 3)

    if graph:
        gs.tight_layout(graph)
        pdf.savefig(graph)
        plt.close('graph')
        graph_location = 0
        graph = plt.figure()
        gs  = gridspec.GridSpec(2, 3)
out_file = image_dir + '/' + output_dir + 'MS2cp-signaldata_2dgaussfit.txt';
outdf.to_csv(out_file, sep='\t', na_rep='NA', index=False);
pdf.close;

## Annotate mp3 files of image data if desired
if animate:
    if not os.path.exists('./Movies'):
        os.mkdir('./Movies')
    os.chdir('./Movies/')
    print('Generating annotated movie outputs....')
    for i in image_list:
        cur_image = i;
        print('\tWorking on image:  ' + cur_image);
        im = io.imread(image_dir + cur_image);
        im = pyM.orderImageDim(im, 1, 'xyt')
        image_spec = pyM.getImageShape(im, 1, 'xyt')
        data = im[:,:, frame_number];

        m = re.search('(Batch\d+_\d+_XY\d+).*.tif', cur_image);
        match = m.group(1);

        loci = outdf[outdf.Image.str.contains(match)];
        if not loci.empty:
            uloci = loci.Image.unique();
            num_loci = uloci.size;
            locus_ID = [None] * num_loci
            frame_number = 0;
            roi = []
            labels = []
            locus_loc_y = np.zeros(num_loci);
            locus_loc_x = np.zeros(num_loci);
            min_pix = np.min(im[np.nonzero(im)])
            
            ###Set max pixel, ignoring very bright hot pixels
            max_pixels = np.zeros(image_spec.frames)
            min_pixels = np.zeros(image_spec.frames)
            for f in range(0, image_spec.frames):
                min_pixels[f] = np.min(im[:, :, f])
                max_pixels[f] = np.max(im[:, :, f])
            max_pix = np.median(max_pixels)
            min_pix = np.median(min_pixels)

            ##Initialize Figure for Animation
            fig = plt.figure(figsize=(5,5), frameon=False);
            ax = fig.add_axes([0,0,1,1], xlim=(0,image_spec.x_size), ylim=(image_spec.y_size,0), aspect='equal', frameon=False);
            ax.set_axis_off();
            plt.margins(0,0)
            ax.set_frame_on(False);

            implt = ax.imshow(data, cmap=plt.cm.gray);
            for n in range(0, num_loci):
                m = re.search('Batch\d+_\d+_XY\d+_(\w+)', uloci[n])
                locus_ID[n] = m.group(1)
                locus_loc_y[n] = outdf.loc[(outdf['Image'] == uloci[n]) & (outdf['T Value (frame)'] == (frame_number + 1)),'Y Value (pixel)'].values[0];
                locus_loc_x[n] = outdf.loc[(outdf['Image'] == uloci[n]) & (outdf['T Value (frame)'] == (frame_number + 1)),'X Value (pixel)'].values[0];
                (local_x_min, local_x_max, local_y_min, local_y_max) = getBounds(locus_loc_x[n], locus_loc_y[n], local_halfwidth, local_size, image_spec)
                
                temp_roi = ax.add_patch(patches.Rectangle((local_x_min, local_y_min), local_size, local_size, fill=False, ls='dotted', ec='yellow')); ###Modified
                roi.append(temp_roi)
                temp_label = ax.text(x=(local_x_min + 3), y=(local_y_min - 3), s=locus_ID[n], color='y', fontsize=6)
                labels.append(temp_label)

            if use_marker:
                scat = ax.scatter([], [], marker='o', facecolors = 'none', edgecolors='r', s=40);
            if stamp:
                mins, sec = divmod((frame_number * time_interval), 60)
                #frame_string = "Min " + str(mins) + " Sec " +str(sec).zfill(2); #This is clunky version of annotation
                frame_string = str(mins) + ":00"; #This version skips annotating every frame but looks nicer in movies.
                frame_text = ax.text(x=10, y=30, s=frame_string, fontsize=16, weight='bold', color='white', fontname='Verdana');
            if scalebar != 0:
                pixel_len = scalebar / pixel_size;
                ax.hlines(10, (500 - pixel_len), 500, color='w')

            animation = FuncAnimation(fig, update, frames=image_spec.frames, repeat=False);
            animation.save(cur_image + '.mp4', writer='ffmpeg', dpi=300, bitrate=24000, fps=14);
            plt.close('all');
