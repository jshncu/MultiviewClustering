function BBC_Demos()
	datasetname = 'BBCSport';
	filename = sprintf('%s.mat',datasetname);
	load(filename);
	foldname = 'initialization';
	cd Tools/;
	addpath(genpath(pwd));
	cd ..;
if ~exist(foldname, 'dir')
    mkdir(foldname);
end
[~,num_view]=size(X.data);
for v=1:num_view
	DATA{1,v}=X.data{1,v}';
end

[~, nsmp] = size(DATA{1,1});
cluster_num=5;
times_clustering=10;

for i=1:nsmp
	for j=1:cluster_num
		if tag(i,j)==1
			true_labels(i,1)=j;
		end
	end
end

for v = 1:num_view
	DATA{v} = DATA{v}./(repmat(sqrt(sum(DATA{v}.^2,1)),size(DATA{v},1),1)+10e-10);
end

filename=sprintf('initialization/%s_init.mat', datasetname);
if ~exist(filename, 'file')
	init_labels = ceil(cluster_num .* rand(nsmp, times_clustering));
	save(filename, 'init_labels');
else
	load(filename);
end

layer_set=[40*cluster_num,20*cluster_num,10*cluster_num,5*cluster_num,cluster_num;];%%3-layer
lambda1_set=10.^[0];
lambda2_set=10.^[0];
lambda3_set=10.^[4];
p_set=[0.1];
AutoNMF_Demos(datasetname, DATA,true_labels,cluster_num,layer_set,lambda1_set,lambda2_set,lambda3_set,p_set)


end
