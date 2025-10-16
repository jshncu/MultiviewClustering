function DANMF_Demos(datasetname, X,true_labels,cluster_num,layer_set,lambda_set,alpha_set,beta_set,p_set)

method = 'DANMF';
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

[~,numOfView]=size(X);
times_clustering = 10;

NMI = zeros(times_clustering, 1);
Acc = zeros(size(NMI));
Purity = zeros(size(NMI));

tolfun_pre = 1e-8;
maxiter_pre = 400;
num_of_layers=3;
[~,n]=size(X{1,1});%%%注意X得是m*n的



for i=1:size(layer_set,1)
    rank_layers=layer_set(i,:);
    for j = 1 : times_clustering
   
        filename = sprintf('initialization/%s_%s_%s_%d_layer1=%d_layer2=%d_layer3=%d.mat', datasetname,inPara.inoptions, method,j,rank_layers(1),rank_layers(2),rank_layers(3));
        if exist(filename, 'file')
          continue
        end
    	fprintf('initialization: %s\n', filename);
    	W = cell(numOfView, num_of_layers);
        H = cell(numOfView, num_of_layers);
    	U = cell(numOfView, num_of_layers);
       

        for v=1:numOfView
           
        	for i_layer = 1:num_of_layers
        		if i_layer == 1
        			AA = X{v}; 
        		else
        			AA = H{v,i_layer - 1};
                end
                [U{v,i_layer}, H{v,i_layer}, ~] = ShallowNMF(AA, rank_layers(i_layer), maxiter_pre, tolfun_pre);
            end   
        end

        for v=1:numOfView
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
      	filename = sprintf('initialization/%s_%s_%s_%d_layer1=%d_layer2=%d_layer3=%d.mat', datasetname,inPara.inoptions, method,j,rank_layers(1),rank_layers(2),rank_layers(3));
      	
    	load(filename);
        
        WW=cell(numOfView,num_of_layers);
        HH=cell(numOfView,num_of_layers);
        UU=cell(numOfView,num_of_layers);
      
    
        for v=1: numOfView
            for layer=1:num_of_layers
                WW{v,layer}=W{v,layer};
                HH{v,layer}=H{v,layer};
                UU{v,layer}=U{v,layer};
               
            end

        end
  
        WCell{1,j} = WW;
        HCell{1,j} = HH;
        UCell{1,j} = UU;
       
    end

    for lambda=lambda_set
    for alpha=alpha_set
    for beta=beta_set
    for p=p_set
    			
        filename1 = sprintf('%s/%s/rank_layers=%d_%d_%d_lambda=%f_alpha=%f_beta=%f_p=%f.mat', ...
            foldname,'DANMF',  rank_layers(1),rank_layers(2),rank_layers(3),lambda,alpha,beta,p);
	 
        if exist(filename1, 'file') 
          continue;
        end

        for j=1 :times_clustering
            fprintf('%s on %s with layer1: %d, layer2: %d, layer3: %d,  lambda: %f, alpha: %f, beta: %f, p: %f\n, j: %d\n', method,datasetname, rank_layers(1),rank_layers(2),rank_layers(3),lambda,alpha,beta,p,j);
         
            options.lambda=lambda;
            options.alpha = alpha;
            options.beta=beta;
            options.p=p;
            options.rho=1.3;
            options.mu=1e-5;
            options.max_mu=10e12;
            options.maxiter=80;
            options.tolfun=1e-13;
            options.eta1=1e-5;
            options.max_eta1=10e12;
           
            [S,H]= DANMF(X, WCell{1,j},UCell{1,j},HCell{1,j}, rank_layers, options);
          
            Vm_sum1=0;
            for v=1:numOfView
                Vm_sum1=Vm_sum1+S{v};
            end
            Final_V1=Vm_sum1./numOfView;
			
            results1 = SpectralClustering(Final_V1,cluster_num);
            [Acc(j,1),NMI(j,1),Purity(j,1)]=ClusteringMeasure(true_labels, results1);
            fprintf('NMI: %f, Acc: %f, Purity: %f\n', NMI(j, 1), Acc(j, 1),Purity(j,1));		
	
        end
	
	    filename1 = sprintf('%s/%s/rank_layers=%d_%d_%d_lambda=%f_alpha=%f_beta=%f_p=%f.mat', ...
            foldname,'DANMF',  rank_layers(1),rank_layers(2),rank_layers(3),lambda,alpha,beta,p);

		if ~exist(filename1, 'file')
			save(filename1, 'NMI', 'Acc','Purity');
		end

    end
    end
    end
    end
end
end
