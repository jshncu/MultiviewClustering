function [labels, vObjValues]= myKmeansClustering(U,numclass,labels)

% rng(seed, 'twister');
U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1,size(U,2));

% indx = litekmeans(U_normalized,numclass,'MaxIter',100, 'Replicates',10);
% indx = indx(:);
[vObjValues, labels, P] = Kmeans(U_normalized', numclass, labels, 100);
labels = labels';
end
