%=========================================================================%
% Set up source and destination vectors in x and y for est_homography     %
%=========================================================================%

x_src = zeros(4,1);
y_src = zeros(4,1);
x_dst = zeros(4,1);
y_dst = zeros(4,1);

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

%{
%-------DEBUG-------
shapeInserter = vision.ShapeInserter('BorderColor', 'Custom');
I_faces_ref = step(shapeInserter, I_ref, int32(bbox_face_ref));
bbox_noses_ref = zeros(size(bbox_face_ref));
for j=1:size(bbox_nose_ref,1)
    bbox_noses_ref(n,:) = bbox_nose_ref(j,:) + [bbox_face_ref(n,1:2)-1 0 0];
    bbox_noses_ref = bbox_noses_ref / ref_scale;
end
shapeInserter = vision.ShapeInserter('BorderColor', 'Custom', 'CustomBorderColor', [0 255 0]);
I_faces_ref = step(shapeInserter, I_faces_ref, int32(bbox_noses_ref(1,:)));
bbox_mouths_ref = zeros(size(bbox_face_ref));
for j=1:size(bbox_mouth_ref,1)
    bbox_mouths_ref(n,:) = bbox_mouth_ref(j,:) + [bbox_face_ref(n,1:2)-1 0 0];
    bbox_mouths_ref = bbox_mouths_ref / ref_scale;
end
shapeInserter = vision.ShapeInserter('BorderColor', 'Custom', 'CustomBorderColor', [255 0 0]);
I_faces_ref = step(shapeInserter, I_faces_ref, int32(bbox_mouths_ref(1,:)));
bbox_rEyes_ref = zeros(size(bbox_face_ref));
for j=1:size(bbox_rEye_ref,1)
    bbox_rEyes_ref(n,:) = bbox_rEye_ref(j,:) + [bbox_face_ref(n,1:2)-1 0 0];
    bbox_rEyes_ref = bbox_rEyes_ref / ref_scale;
end
shapeInserter = vision.ShapeInserter('BorderColor', 'Custom', 'CustomBorderColor', [0 0 255]);
I_faces_ref = step(shapeInserter, I_faces_ref, int32(bbox_rEyes_ref(1,:)));
bbox_lEyes_ref = zeros(size(bbox_face_ref));
for j=1:size(bbox_lEye_ref,1)
    bbox_lEyes_ref(n,:) = bbox_lEye_ref(j,:) + [bbox_face_ref(n,1:2)-1 0 0];
    bbox_lEyes_ref = bbox_lEyes_ref / ref_scale;
end
shapeInserter = vision.ShapeInserter('BorderColor', 'Custom', 'CustomBorderColor', [255 0 255]);
I_faces_ref = step(shapeInserter, I_faces_ref, int32(bbox_lEyes_ref(1,:)));
figure, imshow(I_faces_ref);
%-----END DEBUG-----
%}

%=========================================================================%
% Process input images to find correct homography between our             %
% reference face and the destination image                                %
%=========================================================================%

% load easy sample images
easySampleImages = dir('SampleSet\easy\*.jpg');
numEasySampleImages = length(easySampleImages);

for i=1:numEasySampleImages
    
    % read i-th easy image
    currentImage = strcat('SampleSet\easy\', easySampleImages(i).name);
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
    face_size = [bbox_face(1,3) bbox_face(1,4)];
    face_width = min(bbox_face(:,3));
    face_height = min(bbox_face(:,4));
    scale = initial_scale * double(140 / ((face_width + face_height) / 2.0));
    I_small = imfilter(imresize(I, scale),G);
    bbox_face = bbox_face / initial_scale * scale;%step(faceDetector, I_small);
    
    % initialize comped image and blender object
    img_mosaic = double(I)/256.0;
    panoramaView = imref2d([imheight imwidth], [1 imwidth], [1 imheight]);
    blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');
    
    % initialize feature detectors
    disp(ones(1,2)*ceil(bbox_face(1,3)));
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
        end
        x_dst(1,1) = (bbox_nose(1,1) + (bbox_face(n,1)-1) + 0.5*bbox_nose(1,3)) / scale;
        y_dst(1,1) = (bbox_nose(1,2) + (bbox_face(n,2)-1) + 0.5*bbox_nose(1,4)) / scale;
   
        % detect mouth of face
        bbox_mouth = step(mouthDetector, Icrop);
        j = size(bbox_mouth,1);
        if j==0
            continue;
        end
        lowest_pos = 0; lowest_index = 0;
        for e=1:j
            if (bbox_mouth(e,2) + 0.5*bbox_mouth(e,4) > lowest_pos)
                lowest_pos = bbox_mouth(e,2) + 0.5*bbox_mouth(e,4);
                lowest_index = e;
            end
        end
        x_dst(2,1) = (bbox_mouth(lowest_index,1) + (bbox_face(n,1)-1) + 0.5*bbox_mouth(lowest_index,3)) / scale;
        y_dst(2,1) = (bbox_mouth(lowest_index,2) + (bbox_face(n,2)-1) + 0.5*bbox_mouth(lowest_index,4)) / scale;
    
        % detect right eye of face
        bbox_rEye = step(rEyeDetector, Icrop);
        j = size(bbox_rEye,1); 
        if j==0
            continue;
        end
        most_right_pos = 0; most_right_index = 0;
        for e=1:j
            if (bbox_rEye(e,1) + 0.5*bbox_rEye(e,3) > most_right_pos)
                most_right_pos = bbox_rEye(e,1) + 0.5*bbox_rEye(e,3);
                most_right_index = e;
            end
        end
        x_dst(3,1) = (bbox_rEye(most_right_index,1) + (bbox_face(n,1)-1) + 0.5*bbox_rEye(most_right_index,3)) / scale;
        y_dst(3,1) = (bbox_rEye(most_right_index,2) + (bbox_face(n,2)-1) + 0.5*bbox_rEye(most_right_index,4)) / scale;
    
        % detect left eye of face
        bbox_lEye = step(lEyeDetector, Icrop);
        j = size(bbox_lEye,1); 
        if j==0
            continue;
        end
        most_left_pos = size(Icrop,2); most_left_index = 0;
        for e=1:j
            if (bbox_lEye(e,1) + 0.5*bbox_lEye(e,3) < most_left_pos)
                most_left_pos = bbox_lEye(e,1) + 0.5*bbox_lEye(e,3);
                most_left_index = e;
            end
        end
        x_dst(4,1) = (bbox_lEye(most_left_index,1) + (bbox_face(n,1)-1) + 0.5*bbox_lEye(most_left_index,3)) / scale;
        y_dst(4,1) = (bbox_lEye(most_left_index,2) + (bbox_face(n,2)-1) + 0.5*bbox_lEye(most_left_index,4)) / scale;
    
        % assert right and left eye are different
        if (abs(x_dst(4,1)-x_dst(3,1)) < (size(Icrop,1)/4.0))
            continue;
        end
        
        % Compute homography H (exact)
        % Code for this was borrowed from the Project3 page.
        this_H = est_homography(x_dst, y_dst, x_src, y_src);
        tform = projective2d(this_H');
        [xLimits, yLimits] = outputLimits(tform, [1 2*size(face_ref,2)], [1 2*size(face_ref,1)]);

        % Overlay the warped reference face image onto the destination
        face_ref = im2double(imcrop(I_ref, bbox_face_ref(1,:)/ref_scale));
        %[xLimits, yLimits] = outputLimits(tform, [1 size(face_ref,2)], [1 size(face_ref,1)]);
        %if (max(xLimits)-min(xLimits) > size(img_mosaic,2) || max(yLimits)-min(yLimits) > size(img_mosaic,1))
        %    continue
        %end
        warpedImage = imwarp(face_ref, tform, 'OutputView', panoramaView);
        warpedMask = imcrop(rgb2gray(imread('reference_mask2.png')),bbox_face_ref(1,:)/ref_scale);
        warpedMask = imwarp(warpedMask, tform, 'OutputView', panoramaView) >= 1;
        %warpedMask = warpedMask >= 1;
        img_mosaic = step(blender, img_mosaic, warpedImage, warpedMask);
    end
    
    figure, imshow(img_mosaic);
    imwrite(img_mosaic, 'output_example.bmp');
end

%{
hardSampleImages = dir('SampleSet\hard\*.jpg');
numHardSampleImages = length(hardSampleImages);
for i=1:numEasySampleImages
    currentImage = strcat('SampleSet\hard\', hardSampleImages(i).name);
    I = imread(currentImage);
    I = imresize(I, 0.5);
    bbox = step(faceDetector, I);
    shapeInserter = vision.ShapeInserter('BorderColor', 'Custom');
    I_faces = step(shapeInserter, I, int32(bbox));
    figure, imshow(I_faces);
end
%}