function x = replace_faces(dirName)

fprintf('------------------------------------------\n');
fprintf(strcat('Replacing faces in directory ', dirName, '\n'));

%=========================================================================%
% Set up source and destination vectors in x and y for est_homography     %
%=========================================================================%

x_src = zeros(6,1);
y_src = zeros(6,1);
x_dst = zeros(6,1);
y_dst = zeros(6,1);

%=========================================================================%
% Process reference image featuring face to use as replacement            %
%=========================================================================%

ref_scale = 0.1;

% detect features in reference image
I_ref = imread('reference4.jpg');
I_ref_small = imresize(I_ref, ref_scale);

% detect overall face
faceDetector = vision.CascadeObjectDetector;
bbox_face_ref = step(faceDetector, I_ref_small);
bbox_face_ref = bbox_face_ref(1,:);

% detect nose of face
noseDetector = vision.CascadeObjectDetector('Nose');
for n=1:size(bbox_face_ref,1)
    Icrop = rgb2gray(imcrop(I_ref_small,bbox_face_ref(n,:)));
    bbox_nose_ref = step(noseDetector, Icrop);
    j = size(bbox_nose_ref,1);
    x_src(1,1) = (bbox_nose_ref(1,1) + 0.5*bbox_nose_ref(1,3)) / ref_scale;
    y_src(1,1) = (bbox_nose_ref(1,2) + 0.5*bbox_nose_ref(1,4)) / ref_scale;
end

% detect mouth of face
mouthDetector = vision.CascadeObjectDetector('Mouth');
for n=1:size(bbox_face_ref,1)
    Icrop = rgb2gray(imcrop(I_ref_small,bbox_face_ref(n,:)));
    bbox_mouth_ref = step(mouthDetector, Icrop);
    j = size(bbox_mouth_ref,1);
    x_src(2,1) = (bbox_mouth_ref(j,1) + 0.5*bbox_mouth_ref(j,3)) / ref_scale;
    y_src(2,1) = (bbox_mouth_ref(j,2) + 0.5*bbox_mouth_ref(j,4)) / ref_scale;
end

% detect right eye of face
rEyeDetector = vision.CascadeObjectDetector('RightEye');
for n=1:size(bbox_face_ref,1)
    Icrop = rgb2gray(imcrop(I_ref_small,bbox_face_ref(n,:)));
    bbox_rEye_ref = step(rEyeDetector, Icrop);
    j = size(bbox_rEye_ref,1);
    x_src(3,1) = (bbox_rEye_ref(j,1) + 0.5*bbox_rEye_ref(1,3)) / ref_scale;
    y_src(3,1) = (bbox_rEye_ref(j,2) + 0.5*bbox_rEye_ref(1,4)) / ref_scale;
end

% detect left eye of face
lEyeDetector = vision.CascadeObjectDetector('LeftEye');
for n=1:size(bbox_face_ref,1)
    Icrop = rgb2gray(imcrop(I_ref_small,bbox_face_ref(n,:)));
    bbox_lEye_ref = step(lEyeDetector, Icrop);
    j = size(bbox_lEye_ref,1);
    x_src(4,1) = (bbox_lEye_ref(j,1) + 0.5*bbox_lEye_ref(1,3)) / ref_scale;
    y_src(4,1) = (bbox_lEye_ref(j,2) + 0.5*bbox_lEye_ref(1,4)) / ref_scale;
end

Icrop = rgb2gray(imcrop(I_ref_small,bbox_face_ref(n,:)));
[height, width] = size(Icrop);
x_src(5,1) = 0 / ref_scale; y_src(5,1) = height / ref_scale;
x_src(6,1) = width / ref_scale; y_src(6,1) = height / ref_scale;

%=========================================================================%
% Process input images to find correct homography between our             %
% reference face and the destination image                                %
%=========================================================================%

% load images
inputImages = dir(strcat(dirName, '*.jpg'));
numInputImages = length(inputImages);

for i=1:numInputImages
    
    % read i-th easy image
    currentImage = strcat(dirName, inputImages(i).name);
    fprintf('currentImage = %s\n', currentImage);
    I = imread(currentImage);
    [imheight, imwidth, ~] = size(I);
    
    % establish scale
    initial_scale = 0.5;
    I_small = imresize(I, initial_scale);
    G = fspecial('gaussian', 5, 1.0);
    I_small = imfilter(I_small, G);
    
    % detect overall face and rescale
    bbox_face = step(faceDetector, I_small);
    if (size(bbox_face,1) == 0)
        continue;
    end
    face_width = min(bbox_face(:,3));
    face_height = min(bbox_face(:,4));
    scale = initial_scale * double(140 / ((face_width + face_height) / 2.0));
    I_small = imfilter(imresize(I, scale),G);
    bbox_face = bbox_face / initial_scale * scale;
    
    % initialize comped image and blender object
    img_mosaic = double(I)/256.0;
    panoramaView = imref2d([imheight imwidth], [1 imwidth], [1 imheight]);
    blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');
    
    % initialize feature detectors
    noseDetector = vision.CascadeObjectDetector('Nose', 'MaxSize', ones(1,2)*ceil(bbox_face(1,3)), ...
                                                'ScaleFactor', 1.1, 'MergeThreshold', 10);
    mouthDetector = vision.CascadeObjectDetector('Mouth', 'MaxSize', ones(1,2)*ceil(bbox_face(1,3)), ...
                                                 'ScaleFactor', 1.1, 'MergeThreshold', 10);
    rEyeDetector = vision.CascadeObjectDetector('RightEye', 'MaxSize', ones(1,2)*ceil(bbox_face(1,3)), ...
                                                'ScaleFactor', 1.1, 'MergeThreshold', 10);
    lEyeDetector = vision.CascadeObjectDetector('LeftEye', 'MaxSize', ones(1,2)*ceil(bbox_face(1,3)), ...
                                                'ScaleFactor', 1.1, 'MergeThreshold', 10);
                                                
    % loop through all detected faces in image
    for n=1:size(bbox_face,1)
        
        % isolate face
        Icrop = rgb2gray(imcrop(I_small,bbox_face(n,:)));
        
        % detect nose of face
        bbox_nose = step(noseDetector, Icrop);
        j = size(bbox_nose,1);
        if j==0
            continue;
        else
            x_dst(1,1) = (bbox_nose(1,1) + (bbox_face(n,1)-1) + 0.5*bbox_nose(1,3)) / scale;
            y_dst(1,1) = (bbox_nose(1,2) + (bbox_face(n,2)-1) + 0.5*bbox_nose(1,4)) / scale;
        end
   
        % detect mouth of face
        bbox_mouth = step(mouthDetector, Icrop);
        j = size(bbox_mouth,1);
        if j==0
            continue;
        else
            lowest_pos = 0; lowest_index = 0;
            for e=1:j
                if (bbox_mouth(e,2) + 0.5*bbox_mouth(e,4) > lowest_pos)
                    lowest_pos = bbox_mouth(e,2) + 0.5*bbox_mouth(e,4);
                    lowest_index = e;
                end
            end
            x_dst(2,1) = (bbox_mouth(lowest_index,1) + (bbox_face(n,1)-1) + 0.5*bbox_mouth(lowest_index,3)) / scale;
            y_dst(2,1) = (bbox_mouth(lowest_index,2) + (bbox_face(n,2)-1) + 0.5*bbox_mouth(lowest_index,4)) / scale;
        end
    
        % detect right eye of face
        bbox_rEye = step(rEyeDetector, Icrop);
        j = size(bbox_rEye,1); 
        if j==0
            continue;
        else
            most_right_pos = 0; most_right_index = 0;
            for e=1:j
                if (bbox_rEye(e,1) + 0.5*bbox_rEye(e,3) > most_right_pos)
                    most_right_pos = bbox_rEye(e,1) + 0.5*bbox_rEye(e,3);
                    most_right_index = e;
                end
            end
            x_dst(3,1) = (bbox_rEye(most_right_index,1) + (bbox_face(n,1)-1) + 0.5*bbox_rEye(most_right_index,3)) / scale;
            y_dst(3,1) = (bbox_rEye(most_right_index,2) + (bbox_face(n,2)-1) + 0.5*bbox_rEye(most_right_index,4)) / scale;
        end
    
        % detect left eye of face
        bbox_lEye = step(lEyeDetector, Icrop);
        j = size(bbox_lEye,1); 
        if j==0
            continue;
        else
            most_left_pos = size(Icrop,2); most_left_index = 0;
            for e=1:j
                if (bbox_lEye(e,1) + 0.5*bbox_lEye(e,3) < most_left_pos)
                    most_left_pos = bbox_lEye(e,1) + 0.5*bbox_lEye(e,3);
                    most_left_index = e;
                end
            end
            x_dst(4,1) = (bbox_lEye(most_left_index,1) + (bbox_face(n,1)-1) + 0.5*bbox_lEye(most_left_index,3)) / scale;
            y_dst(4,1) = (bbox_lEye(most_left_index,2) + (bbox_face(n,2)-1) + 0.5*bbox_lEye(most_left_index,4)) / scale;
        end
    
        % add lower bounding box corners of face as additional ctrl pts
        x_dst(5,1) = (bbox_face(n,1)) / scale; y_dst(5,1) = (bbox_face(n,2) + bbox_face(n,4)) / scale;
        x_dst(6,1) = (bbox_face(n,1) + bbox_face(n,3)) / scale; y_dst(6,1) = (bbox_face(n,2) + bbox_face(n,4)) / scale;

        % Compute homography H (exact)
        % Code for this was borrowed from the Project3 page.
        this_H = est_homography(x_dst, y_dst, x_src, y_src);
        tform = projective2d(this_H');

        % DANNY -- USE THIS TO FIND THE CONVEX BOUNDING BOX OF THE
        % TRANSFORMED REFERENCE FACE IN THE SPACE OF THE TARGET IMAGE
        face_ref = im2double(imcrop(I_ref, bbox_face_ref(1,:)/ref_scale));
        [xLimits, yLimits] = outputLimits(tform, [1 bbox_face_ref(1,3)], [1 bbox_face_ref(1,4)]);
        minX = min(xLimits(:)); maxX = max(xLimits(:));
        minY = min(yLimits(:)); maxY = max(yLimits(:));
        
        % Overlay the warped reference face image onto the destination
        warpedImage = imwarp(face_ref, tform, 'OutputView', panoramaView);
        warpedMask = imcrop(rgb2gray(imread('reference_mask2.png')),bbox_face_ref(1,:)/ref_scale);
        warpedMask = imwarp(warpedMask, tform, 'OutputView', panoramaView) >= 1;
        img_mosaic = step(blender, img_mosaic, warpedImage, warpedMask);
    end
    
    figure, imshow(img_mosaic);
    imwrite(img_mosaic, 'output_example.bmp');
end

x = 1;