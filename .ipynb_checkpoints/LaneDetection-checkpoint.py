import cv2
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import moviepy.editor as mpy


mtx = np.array([[1.15777942e+03, 0.00000000e+00, 6.67111050e+02],
       [0.00000000e+00, 1.15282305e+03, 3.86129068e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[-0.24688832, -0.02372817, -0.00109843,  0.00035105, -0.00259133]])



#################################################################################################################################################################################################################################################

#  color based thresholding
def ColorThreshold(im):
    imHLS = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)

    # White-ish areas in image
    # H value can be arbitrary, thus within [0 ... 360] (OpenCV: [0 ... 180])
    # L value must be relatively high (we want high brightness), e.g. within [0.7 ... 1.0] (OpenCV: [0 ... 255])
    # S value must be relatively low (we want low saturation), e.g. within [0.0 ... 0.3] (OpenCV: [0 ... 255])
    white_lower = np.array([np.round(  0 / 2), np.round(0.75 * 255), np.round(0.00 * 255)])
    white_upper = np.array([np.round(360 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])
    white_mask = cv2.inRange(imHLS, white_lower, white_upper)

    # Yellow-ish areas in image
    # H value must be appropriate (see HSL color space), e.g. within [40 ... 60]
    # L value can be arbitrary (we want everything between bright and dark yellow), e.g. within [0.0 ... 1.0]
    # S value must be above some threshold (we want at least some saturation), e.g. within [0.35 ... 1.0]
    yellow_lower = np.array([np.round( 40 / 2), np.round(0.3 * 255), np.round(0.4 * 255)])
    yellow_upper = np.array([np.round( 70 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])
    yellow_mask = cv2.inRange(imHLS, yellow_lower, yellow_upper)

    # Calculate combined mask, and masked image
    colorMask = cv2.bitwise_or(yellow_mask, white_mask)
    ret,colorMask = cv2.threshold(colorMask,0.5,1,cv2.THRESH_BINARY)

#     kernel = np.ones((3,3),np.uint8)
#     colorMask = cv2.dilate(colorMask,kernel,iterations = 1)
#     colorMask = cv2.morphologyEx(colorMask, cv2.MORPH_CLOSE, kernel)
    
    return colorMask

# to get birds eye view
def getPerspectiveTransform(im,inverse=False):
    im_warp = np.copy(im)
    height,width = im.shape[:2]

    a = 150
    horizon = 450
    b = 0.46
    pts = np.array([[a,height],[int(width*b),horizon],[int(width*(1-b)),horizon], [width-a, height]], np.float32)
    res = np.array([[(a+100), height], [(a+100), 0], [width-(a+100), 0], [width-(a+100), height]], np.float32)
    # res = np.array([[a, height], [a, 0], [width-a, 0], [width-a, height]], np.float32) 

    if inverse == True:
        M = cv2.getPerspectiveTransform(res, pts)
    else:   
        M = cv2.getPerspectiveTransform(pts, res)

    im_warp = cv2.warpPerspective(im_warp, M, (width,height), flags=cv2.INTER_LINEAR)
    
    return im_warp

# plot the tracked lane lines in the Map
def drawLaneLines(im_warp,L_line,R_line):

    imlines = im_warp.copy() # get a 
    im_lines = np.dstack((imlines,imlines,imlines))*255
    imlines = cv2.polylines(imlines, [L_line], False,  (255, 100,0), 30)
    imlines = cv2.polylines(imlines, [R_line], False,  (255, 100,0), 30)
    
    return imlines

# Track the lane lines by in bird's eye view image to find trajectory points using centroids
def windowTrace(im_warp, n_windows = 10, W_win = 100):
    
    # define window size parameters
    height,width = im_warp.shape[:2]
    H_win = int(height//n_windows) 


#     half_height = int(height//2) # divide the height by 2    
#     im_warpBOTTOM = im_warpBOTTOM[height:,:] # find bottom half of the warped road (B-map)

    distribution = np.sum(im_warp,axis=0) # plot the pixel distribution along x axis
    mdpt = int(len(distribution)//2) # find midpoint of the distribution alonf x axis    
    Xcleft = np.argmax(distribution[:mdpt]) # find left lane peak as p1
    Xcright = np.argmax(distribution[mdpt:]) + mdpt # find right lane peak as p2 


    # finds all non zero pixels in the Bmap
    Y_nz,X_nz = im_warp.nonzero()

    L_lane_inds = []
    R_lane_inds = []
    
    L_centrds = [(Xcleft,height)]
    R_centrds = [(Xcright,height)]

    # B maps consists of both lane pixels and the distorted back ground content.
    ##  Traverse upwards window-wise to track the lane pixels from the overall B maps
    for win in range(n_windows):
        # define window Y boundaries
        Ywin_start = height - (win+1)*H_win
        Ywin_end = height - (win)*H_win

        # define left window X boundaries 
        Xwin_Lstart, Xwin_Lend = int(Xcleft - W_win//2), int(Xcleft + W_win//2)
        # define right window X boundaries 
        Xwin_Rstart, Xwin_Rend = int(Xcright - W_win//2), int(Xcright + W_win//2)


        '''
        For a given window positions in Bmap Find the nonzero lane pixels within the left and right windows:

        Y_nz and X_nz are list of indices of non zero pixels in our entire B-map. 
        Any non zero pixel in Bmap is present in X_nz[i],Y_nz[i] (i ranges 0-len(X_nz or Y_nz))

        Here, we need to find out the Y_nz and X_nz elements inside the window boundaries.

        Lwin_nz, Rwin_nz initially returns a boolean list. 
        True: indices of nonzero pixels inside the window boundaries. 
        False: indices of nonzero pixels outside window boundaries
        We find the  nonzero() of the boolean list to return indices that are inside the window.
        X_nz(Lwin_nz) gives Bmap non-zero pixel location that are inside the window
        '''

        # Find non zero pixel locations inside the left and right windows
        # here, Lwin_nz and Rwin_nz contains the indices of X_nz that are within the current window
        Lwin_nz = (Y_nz >= Ywin_start)&(Y_nz <= Ywin_end)&(X_nz >= Xwin_Lstart)&(X_nz <= Xwin_Lend) # returns boolean list
        Lwin_nz = Lwin_nz.nonzero()[0] # pick the Bmap nz pixel indices belonging inside window

        Rwin_nz =(Y_nz >= Ywin_start)&(Y_nz <= Ywin_end)&(X_nz >= Xwin_Rstart)&(X_nz <= Xwin_Rend) # return boolean list
        Rwin_nz = Rwin_nz.nonzero()[0] # pick the Bmap nz pixel indices belonging inside window

        # Xaxis updation:
        # the mean of X_nz pixels inside the left and right windows are found. 
        # This mean is the X axis location of the left and right lanes
        # Xcleft and Xcright are updated with the mean location of pixels inside the window
        if len(Rwin_nz)>100:
            Xcright = int(np.mean(X_nz[Rwin_nz]))
        if len(Lwin_nz)>100:
            Xcleft = int(np.mean(X_nz[Lwin_nz]))
 
        L_centrds.append((Xcleft,Ywin_start))
        R_centrds.append((Xcright,Ywin_start))

        # append the list of X_nz[i], Y_nz[i] indices i inside the current left and right windows
        L_lane_inds.append(Lwin_nz)
        R_lane_inds.append(Rwin_nz)
            

    L_lane_inds = np.concatenate(L_lane_inds)
    R_lane_inds = np.concatenate(R_lane_inds)
    # obtain  left and right lane points
    lX,lY = X_nz[L_lane_inds],Y_nz[L_lane_inds]
    rX,rY = X_nz[R_lane_inds],Y_nz[R_lane_inds]
    
    return L_centrds,R_centrds
#     return (lX,lY),(rX,rY)

# get the Polynomial lane line for a bird's eye view map using windowTrace method
def getLaneLines(im_warp):
    height,width = im_warp.shape[:2]
    half_height = int(im_warp.shape[0]//2) # divide the height by 2
    half_width = int(im_warp.shape[1]//2) # divide the height by 2

    im_warpTOP = im_warp[:half_height,:] # find bottom half of the warped road
    im_warpBOTTOM = im_warp[half_height:,:] # find bottom half of the warped road
    im_warpLEFT = im_warp[:,:half_width] # find bottom half of the warped road
    im_warpRIGHT = im_warp[:,half_width:] # find bottom half of the warped road

    L_centrdsTOP,R_centrdsTOP = windowTrace(im_warpTOP,10,100)
    L_centrdsBOTTOM,R_centrdsBOTTOM = windowTrace(im_warpBOTTOM,10,100)

    L_centrdsBOTTOM,R_centrdsBOTTOM = [(i,j+half_height) for (i,j) in L_centrdsBOTTOM],[(i,j+half_height) for (i,j) in R_centrdsBOTTOM]

    L_centrds, R_centrds = np.vstack((np.array(L_centrdsBOTTOM),np.array(L_centrdsTOP[1:]))), np.vstack((np.array(R_centrdsBOTTOM),np.array(R_centrdsTOP[1:])))

    left_curve, right_curve = np.poly1d(np.polyfit(L_centrds[:,1],L_centrds[:,0],3)), np.poly1d(np.polyfit(R_centrds[:,1],R_centrds[:,0],3))

    Laxis,Raxis = np.linspace(L_centrds[:,1].min(),L_centrds[:,1].max()-1), np.linspace(R_centrds[:,1].min(),R_centrds[:,1].max()-1)

    l_fit,r_fit = left_curve(Laxis),right_curve(Raxis)

    L_line, R_line = np.dstack((l_fit.astype(np.int32),Laxis.astype(np.int32))), np.dstack((r_fit.astype(np.int32),Raxis.astype(np.int32)))

    return L_line, R_line

# get the direction of lane from the estimated lane trajectory
def getDirection(L_line,R_line):

    Lanex = (np.squeeze(L_line)[:,0] + np.squeeze(R_line)[:,0])//2
    Laney = (np.squeeze(L_line)[:,1] + np.squeeze(R_line)[:,1])//2

    # Line = np.dstack((Lanex,Laney))

    curveDir = np.poly1d(np.polyfit(Laney,Lanex,1))
    yaxis = np.linspace(Laney.min(),Laney.max()-1)
    linefit = curveDir(yaxis)
    line = np.dstack((linefit.astype(np.int32),yaxis.astype(np.int32)))

    slope = (linefit[-1] - linefit[0]) / (yaxis[-1] - yaxis[0])

    if  slope > 0.1:
        direction = 'left'
    elif  (slope > 0.01) & (slope < 0.1):
        direction = 'slight left'

    elif slope < -0.1 :
        direction = 'right'
    elif  (slope > -0.1) & (slope < -0.01):
        direction = 'slight right'

    else:
        direction = 'straight'
        
#     print(slope,direction)
    return line


def processLane(im):

    im_binary = ColorThreshold(im)  #get the yellow and white lanes

    im_warp = getPerspectiveTransform(im_binary) # get the bird's eye view of the road - denoted as Bmap

    height,width = im_warp.shape[:2]
    half_height = int(im_warp.shape[0]//2) # divide the height by 2
    half_width = int(im_warp.shape[1]//2) # divide the height by 2

    im_warpTOP = im_warp[:half_height,:] # find bottom half of the Bmap
    im_warpBOTTOM = im_warp[half_height:,:] # find bottom half of the Bmap

    # Split the Bmap to upper and lower halves and Find the centroids points by traversing a sliding window in the lane curve.  
    L_centrdsTOP,R_centrdsTOP = windowTrace(im_warpTOP,10,200)
    L_centrdsBOTTOM,R_centrdsBOTTOM = windowTrace(im_warpBOTTOM,10,200)

    # The split resulted in a Y height of (0-360) for bottom Bmap, instead of (360-720). compensate the bottom range by adding an offset of 360 to Y axis
    L_centrdsBOTTOM,R_centrdsBOTTOM = [(i,j+half_height) for (i,j) in L_centrdsBOTTOM],[(i,j+half_height) for (i,j) in R_centrdsBOTTOM]

    # Append the top and bottom centroids to get the total list of centroids of the entire Bmap
    L_centrds, R_centrds = np.vstack((np.array(L_centrdsBOTTOM),np.array(L_centrdsTOP[1:]))), np.vstack((np.array(R_centrdsBOTTOM),np.array(R_centrdsTOP[1:])))

    # get 3rd degree polynomial coefficients for a line fit through the centroid points along the Y axis ie {np.polyfit(Y,X)}
    left_curve, right_curve = np.poly1d(np.polyfit(L_centrds[:,1],L_centrds[:,0],3)), np.poly1d(np.polyfit(R_centrds[:,1],R_centrds[:,0],3))
    # specify the linear range/space for the polynomial line
    Laxis,Raxis = np.linspace(L_centrds[:,1].min(),L_centrds[:,1].max()-1), np.linspace(R_centrds[:,1].min(),R_centrds[:,1].max()-1)
    # fit the polynomial coefficents with the linear space
    l_fit,r_fit = left_curve(Laxis),right_curve(Raxis)


    L_line, R_line = np.dstack((l_fit.astype(np.int32),Laxis.astype(np.int32))), np.dstack((r_fit.astype(np.int32),Raxis.astype(np.int32)))


    # Visualisation works 

    line = getDirection(L_line,R_line) # get the center line with mean of left and right lanes.
    imLines = drawLaneLines(im_warp,L_line,R_line) # draw the estimated lane line 

    inv_imLines1 = getPerspectiveTransform(imLines,True)# get the unwarped estimated lane line 
    inv_im_warp = getPerspectiveTransform(im_warp,True) # get actual unwarped lanes

    # get center Line
    imL = np.dstack((imLines,imLines,imLines))
    imL = cv2.polylines(imL, [line], False,  (0,200, 50), 30)
    ## get lane area in green


    imLines = np.dstack((imLines,imLines,imLines))
    l1,r1 = L_line[0][35:],np.flipud(R_line[0][35:]) # get the last 15 points of left and right lines in right order
    pts = np.vstack((l1,r1)) # stack then in correct order
    imColorLane = cv2.fillPoly(imLines, [pts], (80,227, 227)) #get ColorLanes

    inv_imColorLane = getPerspectiveTransform(imColorLane,True)

    inv_imColorLane = cv2.addWeighted(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 1, inv_imColorLane, 0.5, 0)

    return inv_imColorLane

def ProcessLaneVideo(vidtype, path ="/home/gokul/Projects/LaneDetection/Input/"):
    outputs = []
    count=0
    if vidtype == 'Challenge':
        video = path + 'challenge_video.mp4'
    elif vidtype == 'Project':
        video = path + 'project_video.mp4'
    else:
        print('Improper video type')
    cap = cv2.VideoCapture(video)
    print('Start')
    while(True):
        count+=1 
        ret, frame = cap.read()
        if ret:
            print(count,'----;---', len(outputs))
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame =  cv2.undistort(frame, mtx, dist, None, mtx)
            outputs.append(processLane(frame))
        else:
            break
    cap.release()
    outVideo = mpy.ImageSequenceClip(outputs, fps=25)
    
    return outVideo

#################################################################################################################################################################################################################################################

outVideo = ProcessLaneVideo('Project')
outVideo.write_videofile('output.mp4')
