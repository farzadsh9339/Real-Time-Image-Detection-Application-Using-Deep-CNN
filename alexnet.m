clc;clear all;
tic
convnet = alexnet;
convnet.Layers % Take a look at the layers
rootFolder = 'original_res_train';
categories = {'one','two', 'three', 'four', 'five'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @readFunctionTrain;
[trainingSet, ~] = splitEachLabel(imds, 560, 'randomize'); 
featureLayer = 'output';
trainingFeatures = activations(convnet, trainingSet, featureLayer);
classifier = fitcnb(trainingFeatures, trainingSet.Labels);
rootFolder = 'original_res_test';
testSet = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
testSet.ReadFcn = @readFunctionTrain;
testFeatures = activations(convnet, testSet, featureLayer);
predictedLabels = predict(classifier, testFeatures);
confMat = confusionmat(testSet.Labels, predictedLabels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))
toc