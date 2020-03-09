

function [ labelsOut ] = kNN(X, k, Xt, Lt)
%KNN Your implementation of the kNN algorithm
%   Inputs:
%               X  - Features to be classified
%               k  - Number of neighbors
%               Xt - Training features
%               LT - Correct labels of each feature vector [1 2 ...]'
%
%   Output:
%               LabelsOut = Vector with the classified labels

test = transpose(X)
train = transpose(Xt)

% calculated distance for all feature in test to each feature in train  
dist = pdist2(test,train,'euclidean')

% Sorted the distances, column wise
[~,nearest] = sort(dist,2)

% selected only K nearest distance indices
nearest = nearest(:,1:k)

% Fetched the true labels for previously calculated indices
labels = Lt(nearest)

% took mode : most frequently occuring label (column wise)
labelsOut  = mode(labels,2);


classes = unique(Lt);

numClasses = length(classes);
 

end