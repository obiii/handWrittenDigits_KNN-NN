function [Wout1,Wout2, trainingError, testError ] = trainMultiLayer(Xtraining,Dtraining,Xtest,Dtest, w1, w2,numIterations, learningRate )
%TRAINMULTILAYER Trains the network (Learning)
%   Inputs:
%               X* - Trainin/test features (matrix)
%               D* - Training/test desired output of net (matrix)
%               V0 - Weights of the output neurons (matrix)
%               W0 - Weights of the output neurons (matrix)
%               numIterations - Number of learning setps (scalar)
%               learningRate - The learningrate (scalar)
%
%   Output:
%               Wout - Weights after training (matrix)
%               Vout - Weights after training (matrix)
%               trainingError - The training error for each iteration
%                               (vector)
%               testError - The test error for each iteration
%                               (vector)

% Initiate variables
trainingError = nan(numIterations+1,1);
testError = nan(numIterations+1,1);
numTraining = size(Xtraining,2);
numTest = size(Xtest,2);
numClasses = size(Dtraining,1) - 1;
Wout1 = w1;
Wout2 = w2;

% Calculate initial error
Ytraining = runMultiLayer(Xtraining, w1, w2);
Ytest = runMultiLayer(Xtest, w1, w2);
trainingError(1) = sum(sum((Ytraining - Dtraining).^2))/(numTraining*numClasses);
testError(1) = sum(sum((Ytest - Dtest).^2))/(numTest*numClasses);

for n = 1:numIterations
    
    [ a2,L, a1] = runMultiLayer(Xtraining, Wout1, Wout2);
    te = sum(sum((a2 - Dtraining).^2))/(numTraining*numClasses);
    
    dz2 = a2 - Dtraining;
    grad_w2 = (1/numTraining)* dz2 * transpose(a1);
    
    w2_unbaised = w2(:,2:size(w2,2));
    a1_unbaised = a1(2:end,:);
    dz1 = (transpose(w2_unbaised)*dz2) .* (1-a1_unbaised.^2);
    grad_w1 = (1/numTraining)*dz1* transpose(Xtraining);

    Wout2 = Wout2 - learningRate * grad_w2; %Take the learning step.
    Wout1 = Wout1 - learningRate * grad_w1; %Take the learning step.
    
    Ytraining = runMultiLayer(Xtraining, Wout1, Wout2);
    Ytest = runMultiLayer(Xtest, Wout1, Wout2);

    trainingError(1+n) = sum(sum((Ytraining - Dtraining).^2))/(numTraining*numClasses);
    testError(1+n) = sum(sum((Ytest - Dtest).^2))/(numTest*numClasses);
end

end
