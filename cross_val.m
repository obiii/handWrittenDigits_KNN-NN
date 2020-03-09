function [opt_k, res] = cross_val(Xt,Lt)

opt_k = [2:15]
res = []

for kItem = 1:length(opt_k)
    
    % Get K    
    k = opt_k(kItem)
    avgAcc = 0
    % Split data 
    for iter = 1:length(Xt)
       
        test = Xt{iter}
        testLabel = Lt{iter}
        train =[]
        trainLabels=[]
        for i = 1 : length(Xt)
            if i == iter
             % do 
            else
                train = [train Xt{i}];
                trainLabels =[trainLabels;Lt{i}];
            end  
        end
        
        %Run knn
        LkNN = kNN(test, k, train, trainLabels);
        cM = calcConfusionMatrix( LkNN, testLabel)
        acc = calcAccuracy(cM) %Run knn

        LkNN = kNN(test, k, train, trainLabels);
        cM = calcConfusionMatrix( LkNN, testLabel)
        acc = calcAccuracy(cM)
        avgAcc = avgAcc+acc
    end
    
    avgAcc = avgAcc/length(Xt)

    res = [res;avgAcc];
    
end

res = res

plot(opt_k,res)
end



    
