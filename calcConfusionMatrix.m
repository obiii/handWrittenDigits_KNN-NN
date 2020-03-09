function [ cM ] = calcConfusionMatrix( Lclass, Ltrue )
classes = unique(Ltrue);
numClasses = length(classes);
cM = zeros(numClasses);

%{
ltrue = [1,1,0,1,0,0,0,1,0,1]
ltrue = reshape(ltrue,[10,1])

lpred = [0,0,0,1,1,0,1,0,1,1]
lpred = reshape(lpred,[10,1])
%}

lpred = Lclass;
ltrue = Ltrue;

% Add your own code here
allCl = unique(Ltrue);
cM = zeros(length(allCl),length(allCl));

for i = 1:size(cM,1)
    
    ind = lpred == i;
    
    for  j = 1:size(cM,2)
        
        trueItemsonInd = ltrue(ind);
        
        cM(i,j) = sum(trueItemsonInd==j);
        
    end
end

%cM = confusionmat(Ltrue,Lclass)



end

