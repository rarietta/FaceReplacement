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

for i=1:1
    
    % read i-th easy image
    currentImage = strcat('SampleSet\easy\', easySampleImages(i).name);
    I = imread(currentImage);
    [imheight, imwidth, ~] = size(I);
    
    % establish scale
    scale = 0.33333;
    I_small = imresize(I, scale);
    
    % detect overall face and rescale
    bbox_face = step(faceDetector, I_small);
    %face_width = bbox_face(1,3);
    %face_height = bbox_face(1,4);
    %scale = scale * double(65 / ((face_width + face_height) / 2.0));
    %I_small = imresize(I, scale);
    %bbox_face = step(faceDetector, I_small);
    
    % detect nose of face
    noseDetector = vision.CascadeObjectDetector('Nose');
    for n=1:size(bbox_face,1)
        Icrop = rgb2gray(imcrop(I_small,bbox_face(n,:)));
        bbox_nose = step(noseDetector, Icrop);
        j = size(bbox_nose,1);
        x_dst(1,1) = (bbox_nose(1,1) + (bbox_face(n,1)-1) + 0.5*bbox_nose(1,3)) / scale;
        y_dst(1,1) = (bbox_nose(1,2) + (bbox_face(n,2)-1) + 0.5*bbox_nose(1,4)) / scale;
    end
    
    % detect mouth of face
    mouthDetector = vision.CascadeObjectDetector('Mouth');
    for n=1:size(bbox_face,1)
        Icrop = rgb2gray(imcrop(I_small,bbox_face(n,:)));
        bbox_mouth = step(mouthDetector, Icrop);
        j = size(bbox_mouth,1);
        x_dst(2,1) = (bbox_mouth(1,1) + (bbox_face(n,1)-1) + 0.5*bbox_mouth(1,3)) / scale;
        y_dst(2,1) = (bbox_mouth(1,2) + (bbox_face(n,2)-1) + 0.5*bbox_mouth(1,4)) / scale;
    end
    
    % detect right eye of face
    rEyeDetector = vision.CascadeObjectDetector('RightEye');
    for n=1:size(bbox_face,1)
        Icrop = rgb2gray(imcrop(I_small,bbox_face(n,:)));
        bbox_rEye = step(rEyeDetector, Icrop);
        j = size(bbox_rEye,1);
        x_dst(3,1) = (bbox_rEye(1,1) + (bbox_face(n,1)-1) + 0.5*bbox_rEye(1,3)) / scale;
        y_dst(3,1) = (bbox_rEye(1,2) + (bbox_face(n,2)-1) + 0.5*bbox_rEye(1,4)) / scale;
    end
    
    % detect left eye of face
    lEyeDetector = vision.CascadeObjectDetector('LeftEye');
    for n=1:size(bbox_face,1)
        Icrop = rgb2gray(imcrop(I_small, bbox_face(n,:)));
        bbox_lEye = step(lEyeDetector, Icrop);
        j = size(bbox_lEye,1);
        x_dst(4,1) = (bbox_lEye(1,1) + (bbox_face(n,1)-1) + 0.5*bbox_lEye(1,3)) / scale;
        y_dst(4,1) = (bbox_lEye(1,2) + (bbox_face(n,2)-1) + 0.5*bbox_lEye(1,4)) / scale;
    end

    %{
    %-------DEBUG-------
    figure, imshow(I_ref);
    viscircles([x_src y_src], ones(4,1)*10);
    figure, imshow(I);
    viscircles([x_dst y_dst], ones(4,1)*10);
    
    shapeInserter = vision.ShapeInserter('BorderColor', 'Custom');
    I_faces = step(shapeInserter, I, int32(bbox_face)*3);
    bbox_noses = zeros(size(bbox_face));
    for j=1:size(bbox_nose,1)
        bbox_noses(n,:) = bbox_nose(j,:) + [bbox_face(n,1:2)-1 0 0];
    end
    shapeInserter = vision.ShapeInserter('BorderColor', 'Custom', 'CustomBorderColor', [0 255 0]);
    I_faces = step(shapeInserter, I_faces, int32(bbox_noses)*3);
    bbox_mouths = zeros(size(bbox_face));
    for j=1:size(bbox_mouth,1)
        bbox_mouths(n,:) = bbox_mouth(j,:) + [bbox_face(n,1:2)-1 0 0];
    end
    shapeInserter = vision.ShapeInserter('BorderColor', 'Custom', 'CustomBorderColor', [255 0 0]);
    I_faces = step(shapeInserter, I_faces, int32(bbox_mouths)*3);
    bbox_rEyes = zeros(size(bbox_face));
    for j=1:size(bbox_rEye,1)
        bbox_rEyes(n,:) = bbox_rEye(j,:) + [bbox_face(n,1:2)-1 0 0];
    end
    shapeInserter = vision.ShapeInserter('BorderColor', 'Custom', 'CustomBorderColor', [0 0 255]);
    I_faces = step(shapeInserter, I_faces, int32(bbox_rEyes)*3);
    bbox_lEyes = zeros(size(bbox_face));
    for j=1:size(bbox_lEye,1)
        bbox_lEyes(n,:) = bbox_lEye(j,:) + [bbox_face(n,1:2)-1 0 0];
    end
    shapeInserter = vision.ShapeInserter('BorderColor', 'Custom', 'CustomBorderColor', [255 0 255]);
    I_faces = step(shapeInserter, I_faces, int32(bbox_lEyes)*3);
    
    %-----END DEBUG-----
    %}
    
    % Compute homography H (exact)
    % Code for this was borrowed from the Project3 page.
    this_H = est_homography(x_dst, y_dst, x_src, y_src);
    tform = projective2d(this_H');
   
    % initialize comped image and blender object
    img_mosaic = double(I)/256.0;
    panoramaView = imref2d([imheight imwidth], [1 imwidth], [1 imheight]);
    blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');
    
    % Overlay the warped reference face image onto the destination
    warpedImage = imwarp(face_ref, tform, 'OutputView', panoramaView);
    warpedMask  = imwarp(ones(size(face_ref(:,:,1))), tform, 'OutputView', panoramaView);
    warpedMask  = warpedMask >= 1;
    img_mosaic = step(blender, img_mosaic, warpedImage, warpedMask);
    figure, imshow(img_mosaic);
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