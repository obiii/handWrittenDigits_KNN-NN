%% This script w1ill help you test out your single layer neural netw1ork code

%% Select w1hich data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 3; % Change this to load new1 data 

[X, D, L] = loadDataSet( dataSetNr );

%% Select a subset of the training features

numBins = 2; % Number of Bins you w1ant to dew2ide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true; % true = select features at random, false = select the first features

[ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

% Note: Xt, Dt, Lt w1ill be cell arrays, to extract a bin from them use i.e.
% XBin1 = Xt{1};
%% Modify the X Matrices so that a bias is added

% The Training Data
bias = ones(1,size(Xt{1}(:,1:10),2))
Xtraining = [bias; Xt{1}(:,1:10)];
%Xtraining = Xt{1}

% The Test Data
bias = ones(1,size(Xt{2},2))
Xtest = [bias; Xt{2}];
%Xtest = Xt{2}

%% Train your single layer netw1ork
% Note: You nned to modify trainSingleLayer() in order to train the netw1ork

numHidden = 22; % Change this, Number of hidde neurons 
numIterations = 12000; % Change this, Numner of iterations (Epochs)
learningRate = 0.005; % Change this, Your learningrate

w1= unifrnd(-0.1, 0.1*ones(numHidden ,size(Xtraining,1))); % Change this, Initiate your w1eight matrix w1
w2 = unifrnd(-0.1, 0.1*ones(size(Dt{1}, 1) , numHidden +1)); % Change this, Initiate your w1eight matrix w2

%
tic
[w1,w2, trainingError, testError ] = trainMultiLayer(Xtraining,Dt{1}(:,1:10),Xtest,Dt{2}, w1,w2,numIterations, learningRate );
trainingTime = toc;
%% Plot errors
figure(1101)
clf
[mErr, mErrInd] = min(testError);
plot(trainingError,'k','linewidth',1.5)
hold on
plot(testError,'r','linewidth',1.5)
plot(mErrInd,mErr,'bo','linewidth',1.5)
hold off
title('Training and Test Errors, Multi-Layer')
legend('Training Error','Test Error','Min Test Error')

%% Calculate The Confusion Matrix and the Accuracy of the Ew2aluation Data
% Note: you haw2e to modify the calcConfusionMatrix() function yourselfs.

[ Y, LMultiLayerTraining ] = runMultiLayer(Xtraining, w1, w2);
tic
[ Y, LMultiLayerTest ] = runMultiLayer(Xtest, w1,w2);
classificationTime = toc/length(Xtest);
% The confucionMatrix
cM = calcConfusionMatrix( LMultiLayerTest, Lt{2})

% The accuracy
acc = calcAccuracy(cM);

display(['Time spent training: ' num2str(trainingTime) ' sec'])
display(['Time spent calssifying 1 feature w2ector: ' num2str(classificationTime) ' sec'])
display(['Accuracy: ' num2str(acc)])

%% Plot classifications
% Note: You do not need to change this code.

if dataSetNr < 4
    plotResultMultiLayer(w1,w2,Xtraining,Lt{1},LMultiLayerTraining,Xtest,Lt{2},LMultiLayerTest)
else
    plotResultsOCR( Xtest, Lt{2}, LMultiLayerTest )
end
