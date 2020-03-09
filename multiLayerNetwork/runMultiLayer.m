function [ a2,L,a1] = runMultiLayer( X, w1, w2 )
%RUNMULTILAYER Calculates output and labels of the net
%   Inputs:
%               X  - Features to be classified (matrix)
%               W  - Weights of the hidden neurons (matrix)
%               V  - Weights of the output neurons (matrix)
%
%   Output:
%               Y = Output for each feature, (matrix)
%               L = The resulting label of each feature, (vector) 

% add bias to X

z1 = w1*X;
a1 = tanh(z1);

% add bias to a1
b2 = ones(1,size(a1,2));
a1 = [b2; a1];
z2 = w2*a1;
a2 = tanh(z2);

   

% Calculate classified labels
[~, L] = max(a2,[],1);
L = L(:);

end

