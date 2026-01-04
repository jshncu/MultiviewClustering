function TNAGL_Demos(datasetname, X,true_labels,cluster_num,layer_set,lambda1_set,lambda2_set,lambda3_set,p_set)

method = 'TNAGL';
addpath(sprintf('../%s', method));
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
times_clustering = 1;

NMI = zeros(times_clustering, 1);
Acc = zeros(size(NMI));
Purity = zeros(size(NMI));

tolfun_pre = 1e-5;
maxiter_pre = 100;
num_of_layers=3;
[~,n]=size(X{1,1});%%%注意X得是m*n的

for v=1:num_view
	original_X{1,v}=X{1,v};
end
%m=2000;
m=fix(0.1*n);
for v=1:num_view
	X{1,v}=X{1,v}';
end
opt1. style = 1;
opt1. IterMax =50;
opt1. toy = 0;
%[AA, BB] = My_Bipartite_Con2(X,m,opt1);
[AA, BB] = My_Bipartite_Con(X,cluster_num,0.1,opt1,10);
for v=1:num_view
	B{1,v}=BB{v,1}';
    A{1,v}=AA{v}';
end

for i=1:size(layer_set,1)
    rank_layers=layer_set(i,:);
    for j = 1 : times_clustering
        filename = sprintf('initialization/%s_%s_%s_%d_layer1=%d_layer2=%d_layer3=%d_Anchor=%f.mat', datasetname,inPara.inoptions, method,j,rank_layers(1),rank_layers(2),rank_layers(3),m);
    	if exist(filename, 'file')
         continue
       end
		
    	fprintf('initialization: %s\n', filename);
    	W = cell(num_view, num_of_layers);
        H = cell(num_view, num_of_layers);
    	U = cell(num_view, num_of_layers);
       
        for v=1:num_view
           
        	for i_layer = 1:num_of_layers
        		if i_layer == 1
        			AA = B{v}; 
        		else
        			AA = H{v,i_layer - 1};
                end
                [U{v,i_layer}, H{v,i_layer}, ~] = ShallowNMF777(AA, rank_layers(i_layer), maxiter_pre, tolfun_pre);
                
            end   
        end

        for v=1:num_view
            W{v,1}=U{v,3}';
            W{v,2}=U{v,2}';
            W{v,3}=U{v,1}';
        end
        save(filename, 'W','H','U');
    end
end

k_means_init=sprintf('initialization/%s_init.mat', datasetname);
load(k_means_init);
for i=1:size(layer_set,1)
	rank_layers=layer_set(i,:);
	WCell = cell(1,times_clustering);
	HCell = cell(1,times_clustering);
	UCell = cell(1,times_clustering);
   
  	for j = 1 : times_clustering
      	filename = sprintf('initialization/%s_%s_%s_%d_layer1=%d_layer2=%d_layer3=%d_Anchor=%f.mat', datasetname,inPara.inoptions, method,j,rank_layers(1),rank_layers(2),rank_layers(3),m);
    	load(filename);
        
        WW=cell(num_view,num_of_layers);
        HH=cell(1,num_view);
        UU=cell(num_view,num_of_layers);
   
        for v=1: num_view
            for layer=1:num_of_layers
                WW{layer,v}=W{v,layer};
                HH{layer,v}=H{v,layer};
                UU{layer,v}=U{v,layer};
            end
        end
		
        WCell{1,j} = WW;
        HCell{1,j} = HH;
        UCell{1,j} = UU;
       
    end
   
    for lambda1=lambda1_set
    for lambda2=lambda2_set
	for lambda3=lambda3_set
	for p=p_set
    			
        filename = sprintf('%s/%s/rank_layers=%d_%d_%d_lambda1=%f_lambda2=%f_lambda3=%f_p=%f_m=%d.mat', ...
            foldname,'TNAGL',rank_layers(1),rank_layers(2),rank_layers(3),lambda1,lambda2,lambda3,p,m);

        if exist(filename, 'file')
            continue;
        end

        for j=1 :times_clustering
            fprintf('%s on %s with layer1: %d, layer2: %d, layer3: %d, lambda1: %f, lambda2: %f, lambda3: %f, p: %f, m: %d, j: %d\n', method,datasetname, rank_layers(1),rank_layers(2),rank_layers(3),lambda1,lambda2,lambda3,p,m,j);
            options.lambda1=lambda1;
            options.lambda2=lambda2;
			options.lambda3=lambda3;
	        options.p=p;
			options.num_view=num_view;
		    options.num_layer=num_of_layers;
			options.n=n;
            options.rho1=1.7;
			options.rho2=1.7;
			options.rho3=1.7;
            options.mu1=1e-9;
            options.max_mu1=10e10;
            options.mu2=1e-9;
            options.max_mu2=10e10;
			options.mu3=1e-9;
            options.max_mu3=10e10;
			options.maxiter=100;
            options.tolfun=1e-13;
           
          %  tic;
            [C_Tensor]=TNAGL(original_X,B,A, WCell{1,j},UCell{1,j},HCell{1,j}, rank_layers, options);
            %elapsed_time=toc;
		    %disp(['代码执行时间为：', num2str(elapsed_time), '秒。']);
       
            C=cell(1,num_view);
            for v=1:num_view
                C{1,v}=C_Tensor(:,:,v);
            end
            C_sum=0;
            for v=1:num_view
                C_sum=C_sum+C{1,v};
            end

            Final_V1=C_sum./num_view;
            Final_V1_normalized = Final_V1 ./ repmat(sqrt(sum(Final_V1.^2, 2)), 1,size(Final_V1,2));

            for jj=1:1
                [~, results3] = Kmeans(Final_V1_normalized, cluster_num, init_labels(:, jj)', 1000);
                [NMI(jj,1),Acc(jj,1),Purity(jj,1)]= myClusteringMeasure(true_labels, results3);
            end
            fprintf('NMI: %f, Acc: %f, Purity: %f\n', mean(NMI), mean(Acc),mean(Purity));
        end
        filename = sprintf('%s/%s/rank_layers=%d_%d_%d_lambda1=%f_lambda2=%f_lambda3=%f_p=%f_m=%d.mat', ...
            foldname,'TNAGL',rank_layers(1),rank_layers(2),rank_layers(3),lambda1,lambda2,lambda3,p,m);

        if ~exist(filename, 'file')
            save(filename, 'NMI', 'Acc','Purity');
        end
    end
    end
    end
    end
    end
end


