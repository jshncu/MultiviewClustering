function [NMI, Acc, Purity, Fscore, Precision] = myClusteringMeasure(G, datasetname, true_labels, times_clustering)

cluster_num = length(unique(true_labels));
filename = sprintf('init_%s.mat', datasetname);
if exist(filename, 'file')
    load(filename);
else
    % initial clustering labels for k-means
    init_labels = ceil(length(unique(true_labels)) .* rand(length(true_labels), times_clustering));
    save(filename, 'init_labels');
end
% NMI = zeros(times_clustering, 1);
% Purity = zeros(size(NMI));
% Acc = zeros(size(NMI));
% Fscore = zeros(size(NMI));
% Precision = zeros(size(NMI));
% for i = 1 : times_clustering
	% [results, obj] = myKmeansClustering(G, cluster_num, init_labels(:, i));  

	% NMI(i) = nmi(true_labels,results);
	% conf_mat = confusionmat(true_labels,results);
	% purity = sum(max(conf_mat',[],2)) / length(true_labels);
	% Purity(i) = purity; 
	% Acc(i) = ClusteringAccuracy(true_labels, results); %ok
	% [fscore,precision,~] = compute_f(true_labels,results);
	% Fscore(i) = fscore;
	% Precision(i) = precision;
% end
obj = realmax;
[tmplabel, tmpobj] = myKmeansClustering(G, cluster_num, init_labels(:, 1)); 
results = tmplabel;
for i = 2 : times_clustering
	[tmplabel, tmpobj] = myKmeansClustering(G, cluster_num, init_labels(:, i));  
	if obj > tmpobj
		results = tmplabel;
	end
end
NMI = nmi(true_labels,results);
conf_mat = confusionmat(true_labels,results);
purity = sum(max(conf_mat',[],2)) / length(true_labels);
Purity = purity; 
Acc = ClusteringAccuracy(true_labels, results); %ok
[fscore,precision,~] = compute_f(true_labels,results);
Fscore = fscore;
Precision = precision;
fprintf('NMI: %f, Acc: %f, Purity: %f, Fscore: %f,Precision:%f\n', mean(NMI), mean(Acc), mean(Purity), mean(Fscore), mean(Precision));
end