function Reuters_Demos()
clc;
clear;
close all;
addpath('Tools');
datasetname = 'Reuters';
foldname = 'datasets';
filename = sprintf('%s/%s', foldname, datasetname);
load(filename);

[num_view]=length(fea);
for v=1:num_view
	DATA{1,v}=double(fea{v}');
end

[~, nsmp] = size(DATA{1,1});
cluster_num=length(unique(Y));
times_clustering=1;

true_labels = (double(Y + 1));


%%归一化
%for v=1:num_view
%DATA{1,v} = NormalizeData(DATA{1,v},2);
%end
for v=1:num_view
%DATA{1,v} = bsxfun(@rdivide,DATA{1,v},sqrt(sum(DATA{1,v}.^2,1)));
	size(DATA{1, v})
	DATA{1,v} = NormalizeFea(DATA{1,v},0);
%DATA{1,v} = NormalizeData(DATA{1,v}, 2);
end
num_view
rng(9, 'twister');
MultiEvalutions2(datasetname,DATA,true_labels,cluster_num,909);
end
