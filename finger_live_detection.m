clc;clear all;
tic
% IMAGES ARE CONVERTED INTO 128*128
rootFolderTrain = 'training_data' ;
categories = {'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'};
imdsTrain = imageDatastore(fullfile(rootFolderTrain, categories), 'LabelSource', 'foldernames');
imageSize = [128 128 3];
layers = [
    imageInputLayer(imageSize)
    convolution2dLayer(5,64)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(5,64)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(384)
    fullyConnectedLayer(192)
    fullyConnectedLayer(5)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 100, ...
    'Verbose', true, 'Plots','training-progress');

% Train the network

cifar10Net = trainNetwork(imdsTrain, layers, options );

% TO AVOID TRAINING, YOU CAN LOAD TRAINED MODEL
% load('final_train.mat');
%% KEEP IN MIND THAT PICTURES CAPTURED WITH CONSTANT BACKGROUND SO WHILE DETECTING THERE SHOULD BE CONSTANT BACKGROUND
camera = webcam;
h = figure;
h.Position(3) = 2*h.Position(3);
ax1 = subplot(1,2,1);
ax2 = subplot(1,2,2);
ax2.ActivePositionProperty = 'position';
keepRolling = true;
set(gcf,'CloseRequestFcn','keepRolling = false; closereq');

while keepRolling
    % Display and classify the image
    im = snapshot(camera);
    image(ax1,im)
    im = imresize(im, [128 128]);
    [label,score] = classify(cifar10Net,im);
    title(ax1,{char(label),num2str(max(score),2)});
    
    % Select the top five predictions
    classNames = cifar10Net.Layers(end).ClassNames
    [~,idx] = sort(score,'descend');
    idx = idx(5:-1:1);
    scoreTop = score(idx);
    classNamesTop = classNames(idx);
    
    % Plot the histogram
    barh(ax2,scoreTop)
    title(ax2,'Top 5')
    xlabel(ax2,'Probability')
    xlim(ax2,[0 1])
    yticklabels(ax2,classNamesTop)
    ax2.YAxisLocation = 'right';
    drawnow
end
toc