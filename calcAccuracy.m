function [ acc ] = calcAccuracy( cM )
%CALCACCURACY Takes a confusion matrix amd calculates the accuracy

acc = 0; % Replace with your own code

acc = sum(diag(cM))/sum(sum(cM))

end

