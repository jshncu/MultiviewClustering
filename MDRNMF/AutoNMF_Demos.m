function newNMF_Demos(datasetname, X,true_labels,cluster_num,layer_set,lambda1_set,lambda2_set,lambda3_set,p_set)

method = 'AutoNMF';
%addpath(sprintf('../%s', method));
foldname = '../results';
if ~exist(foldname, 'dir')
    mkdir(foldname);
end
foldname = sprintf('%s/%s', foldname, datasetname);
if ~exist(foldname, 'dir')
    mkdir(foldname);
end
inPara.inoptions = 'NMF';
foldname = sprintf('%s/%s', foldname, method);
if ~exist(foldname, 'dir')
    mkdir(foldname);
end
foldname = sprintf('../results/%s', datasetname);

[~,num_view]=size(X);
times_clustering = 10;


NMI7 = zeros(times_clustering, 1);
Acc7 = zeros(size(NMI7));
Purity7 = zeros(size(NMI7));

NMI8 = zeros(times_clustering, 1);
Acc8 = zeros(size(NMI8));
Purity8 = zeros(size(NMI8));


tolfun_pre = 1e-5;
maxiter_pre = 200;
num_of_layers=3;
[~,n]=size(X{1,1});







for i=1:size(layer_set,1)
rank_layers=layer_set(i,:);
for j = 1 : times_clustering

    filename = sprintf('initialization/%s_%s_%s_%d_layer1=%d_layer2=%d_layer3=%d_layer4=%d_layer5=%d.mat', datasetname,inPara.inoptions, method,j,rank_layers(1),rank_layers(2),rank_layers(3),rank_layers(4),rank_layers(5));
	if exist(filename, 'file')
        continue
    end
	fprintf('initialization: %s\n', filename);
	W = cell(num_view, num_of_layers);
    H = cell(num_view, num_of_layers);
	U = cell(num_view, num_of_layers);
    V = cell(num_view, num_of_layers);
    for v=1:num_view
	for i_layer = 1:num_of_layers
		if i_layer == 1
			ZZ = X{1,v};
			[U{v,i_layer}, H{v,i_layer}] = ShallowNMF(ZZ, rank_layers(i_layer+2), maxiter_pre, tolfun_pre);
			[W{v,i_layer}, V{v,i_layer}] = ShallowNMF(ZZ, rank_layers(i_layer), maxiter_pre, tolfun_pre);
		else
			AA = H{v,i_layer - 1};
			[U{v,i_layer}, H{v,i_layer}] = ShallowNMF(AA, rank_layers(i_layer+2), maxiter_pre, tolfun_pre);
			BB = W{v,i_layer - 1};
			[W{v,i_layer}, V{v,i_layer}] = ShallowNMF(BB, rank_layers(i_layer), maxiter_pre, tolfun_pre);
		end
	end
	U{v,1}=W{v,num_of_layers};
	end
    save(filename, 'W','H','U','V');
end
end




k_means_init=sprintf('initialization/%s_init.mat', datasetname);
load(k_means_init);
for i=1:size(layer_set,1)
	rank_layers=layer_set(i,:);
	WCell = cell(1,times_clustering);
	HCell = cell(1,times_clustering);
	UCell = cell(1,times_clustering);
	VCell = cell(1,times_clustering);
  	for j = 1 : times_clustering
  	filename = sprintf('initialization/%s_%s_%s_%d_layer1=%d_layer2=%d_layer3=%d_layer4=%d_layer5=%d.mat', datasetname,inPara.inoptions, method,j,rank_layers(1),rank_layers(2),rank_layers(3),rank_layers(4),rank_layers(5));
	load(filename);
		WW=cell(num_view,num_of_layers);
		HH=cell(num_view,num_of_layers);
		UU=cell(num_view,num_of_layers);
		VV=cell(num_view,num_of_layers);
		for v=1:num_view
			for layer=1:num_of_layers
				WW{v,layer}=W{v,layer};
				HH{v,layer}=H{v,layer};
				UU{v,layer}=U{v,layer};
				VV{v,layer}=V{v,layer};
			end
		end
		WCell{1,j} = WW;
		HCell{1,j} = HH;
		UCell{1,j} = UU;
		VCell{1,j} = VV;
    end
		for lambda1=lambda1_set
		for lambda2=lambda2_set
		for lambda3=lambda3_set
		for p=p_set
				filename = sprintf('%s/%s/rank_layers=%d_%d_%d_%d_%d_lambda1=%f_lambda2=%f_lambda3=%f_p=%f.mat', ...
					foldname,'AutoNMF7',  rank_layers(1),rank_layers(2),rank_layers(3),rank_layers(4),rank_layers(5),lambda1,lambda2,lambda3,p);
				if exist(filename, 'file')
					continue;
				end
				for j=1 :times_clustering
					fprintf('%s on %s with layer1: %d, layer2: %d, layer3: %d,  layer4: %d,  layer5: %d, lambda1: %f, lambda2: %f, lambda3: %f, p: %f, j: %d\n', method,datasetname, rank_layers(1),rank_layers(2),rank_layers(3),rank_layers(4),rank_layers(5),lambda1,lambda2,lambda3,p,j);
					options.lambda1=lambda1;
					options.lambda2 = lambda2;
					options.lambda3=lambda3;
					options.num_view=num_view;
					options.num_layer=num_of_layers;
					options.n=n;
					options.rho=1.6;
					options.mu=1e-5;
					options.max_mu=10e10;
					options.p=p;
					options.maxiter=100;
					options.tolfun=1e-13;
					[Vm]= AutoNMF(X, WCell{1,j},VCell{1,j},UCell{1,j},HCell{1,j},rank_layers, options);
					
		
					Vm_sum=0;
					for v=1:num_view
						Vm_sum=Vm_sum+Vm{v,num_of_layers};
					end
					Final_V=Vm_sum./num_view;
					[~, results7] = Kmeans(Final_V, cluster_num, init_labels(:, j)', 1000);
					
					
					
					[Acc7(j,1),NMI7(j,1),Purity7(j,1)]=ClusteringMeasure(true_labels, results7);
					
					
					
					%fprintf('NMI: %f, Acc: %f, Purity: %f\n', NMI7(j,1), Acc7(j,1),Purity7(j,1));
				end
				fprintf('NMI: %f, Acc: %f, Purity: %f\n', mean(NMI7), mean(Acc7),mean(Purity7));
		end
		end
		end
		end
end

end

