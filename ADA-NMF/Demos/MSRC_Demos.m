function MSRC_Demos()
	clc;
	clear;
	close all;
	cd ..;
	cd Tools/;
	addpath(genpath(pwd));
        cd ..;
	cd Demos/;
	datasetname = 'MSRC_v1'; 
	filename = sprintf('%s.mat',datasetname);
	load(filename);
	foldname = 'initialization';
	%warning('off')

if ~exist(foldname, 'dir')
    mkdir(foldname);
end
[~,num_view]=size(X.data);
for v=1:num_view
	DATA{1,v}=X.data{1,v}';
end


[~, nsmp] = size(DATA{1,1});
cluster_num=7;
times_clustering=10;

for i=1:nsmp
	for j=1:cluster_num
		if tag(i,j)==1
			true_labels(i,1)=j;
		end
	end
end

for v=1:num_view
DATA{1,v} = bsxfun(@rdivide,DATA{1,v},sqrt(sum(DATA{1,v}.^2,1)));
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
