function UCI2_Demos()
clc;
	clear;
	close all;
	cd ..;
	cd Tools/;
	addpath(genpath(pwd));
    cd ..;
	cd Demos/;
	datasetname = 'UCI2'; 
	filename = sprintf('%s.mat',datasetname);
	load(filename);
	foldname = 'initialization';
	%warning('off')

if ~exist(foldname, 'dir')
    mkdir(foldname);
end

DATA{1, 1} = double(X1);
DATA{1, 2} = double(X2);
DATA{1, 3} = double(X3);
num_view = length(DATA);

[~, nsmp] = size(DATA{1,1});
cluster_num=length(unique(gtCopy));
times_clustering=1;

true_labels = (double(gtCopy));

%%归一化
for v=1:num_view
DATA{v} = DATA{v}./(repmat(sqrt(sum(DATA{v}.^2,1)),size(DATA{v},1),1)+10e-10);
end

filename=sprintf('initialization/%s_init.mat', datasetname);
if ~exist(filename, 'file')
	init_labels = ceil(cluster_num .* rand(nsmp, times_clustering));
	save(filename, 'init_labels');
else
	load(filename);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
end
MultiEvalutions(datasetname,DATA,true_labels,cluster_num);
end
