function Anchors_Demos(datasetname,X,true_labels,cluster_num,m_set,alpha_set,beta_set,gamma_set,k2_set,seed)

method = 'anchors';
addpath(sprintf('%s', method));
foldname = 'results';
if ~exist(foldname, 'dir')
    mkdir(foldname);
end
foldname = sprintf('%s/%s', foldname, datasetname);
if ~exist(foldname, 'dir')
    mkdir(foldname);
end

times_clustering = 10;
num_view = length(X);
NMI = zeros(times_clustering, 1);
Acc = zeros(size(NMI));
Purity = zeros(size(NMI));



[~,n]=size(X{1,1});


for k2=k2_set
	for rp1 = 1:length(m_set)
		m = m_set(rp1);
		X2 = cell(size(X));
		for iv = 1 : num_view
			X2{iv} = X{iv}';
		end
		[centers, B] = FastmultiCLR(X2, k2, k2 * m); % initialize A
		clear X2
		for iv = 1:num_view
			A{iv} = centers{iv}';
			Z{iv} = B{iv}';
		end
		for alpha=alpha_set
			for beta=beta_set
				for gamma = gamma_set
					if k2 * m > n
						continue;
                    end
					fprintf('%s on %s with  m: %f, alpha: %f, beta: %f, gamma: %f, k2: %f, seed: %d\n', method, datasetname, m, alpha, beta, gamma, k2, seed);
					param.alpha = alpha;
					param.beta = beta;
					param.gamma = gamma;
					param.r = 2;
					param.k2 = k2;
					param.maxiter=200;%40
					param.tolfun=1e-6;%1e-13
					begin = tic;
					[G2, G3, obj] = algo_qp(X, m, cluster_num, A, Z, param);
					cost = toc(begin);
					[NMI2, Acc2, Purity2, Fscore2, Precision2] = myClusteringMeasure(G2, datasetname, true_labels, times_clustering);
					[NMI3, Acc3, Purity3, Fscore3, Precision3] = myClusteringMeasure(G3, datasetname, true_labels, times_clustering);
                end
            end
        end
    end
end
end

function [NMI, Acc, Purity, Fscore, Precision] = clusteringresults(labels, true_labels)
results = labels;
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




