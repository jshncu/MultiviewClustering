function MultiEvalutions2(datasetname, XX, true_labels,cluster_num, seed)

%%%%%%%%%%%%%%%%%%%%%%%%%%%anchor%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha_set=10.^[3:-1:-3];
beta_set=10.^[3:-1:-3];
m_set=[9];
k2_set=[1:10] * cluster_num;
r_set=[2];
gamma_set = 10.^[3:-1:-3];

Anchors_Demos(datasetname,XX,true_labels,cluster_num,m_set,alpha_set,beta_set,gamma_set,k2_set, seed);


% % % c_anchor = [m_set, m_set * 2, m_set * 3, m_set * 4, m_set * 5];   % 每个簇包含的锚点数
% % % c_anchor = unique(c_anchor);

% % % %(AAAI2024)Learning Cluster-Wise Anchors for Multi-View Clustering
% alpha_set = [1e-3, 1e-2, 1e-1, 1e0, 1e1];  % alpha
% beta_set = [1e-1, 1e0, 1e1, 1e2, 1e3];  % beta
% c_anchor = [1, 3, 5];
% CAMVC_Demos(datasetname, XX,cluster_num,true_labels,c_anchor,alpha_set,beta_set,seed); % 26 * 5 * 5 = 650

% % % %%(TIP2021)Fast Parameter-Free Multi-View Subspace Clustering With Consensus Anchor Guidance
% numanchor= cluster_num;
% FPMVS_Demos(datasetname, XX,cluster_num,true_labels,numanchor, seed);  % 26

% % % %%%%%%%%%%%%%%%%%%%%%%%%%%2.SMVSC%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % %%(ACM2021)SMVSC-Scalable Multi-view Subspace Clustering with Unified Anchors
% numanchor = [1:3]*cluster_num;	
% SMVSC_Demos(datasetname, XX,cluster_num,true_labels, numanchor, seed); %26



% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%8.LMVSC%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % %%%(AAAI2020)Large-Scale Multi-View Subspace Clustering in Linear Time
% alpha_set = [0.001, 0.01, 0.1, 1, 10];  % alpha
% numanchor = [cluster_num, 50, 100];
% LMVSC_Demos(datasetname, XX,cluster_num,true_labels, numanchor,alpha_set); % 5*26=130 have not to re-evaluate this method

% % % %(TKDE2024)Robust_and_Consistent_Anchor_Graph_Learning_for_Multi-View_Clustering
% lambda_set = [0.1 100 1000 10^6];
% numanchor= [1 : 3] * cluster_num;
% RCAGL_Demos(datasetname, XX,cluster_num,true_labels, numanchor,lambda_set) % 4*26=104  have not to re-evaluate this method

% % % %%(AAAI2022)Efficient One-Pass Multi-View Subspace Clustering with Consensus Anchors
% c_anchor = [1:7];
% EOMSC_Demos(datasetname, XX,cluster_num,true_labels, c_anchor); %7 have not to re-evaluate this method

% % % CVPR 2024 "Learn from View Correlation: An Anchor Enhancement Strategy for Multi-view Clustering"
% c_anchor=[1, 2, 5];
% gamma_set = 10.^[-1:2];
% lambda_set = 10.^[-4:2:2];
% alignPara_set = 10.^[-4, 0, 4];
% AEVC_Demos(datasetname, XX,cluster_num,true_labels,c_anchor, alignPara_set, gamma_set,lambda_set, seed);

% %%%%%%%%%%%%%%%%%%%%%%%%Row Vectors%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_Views = length(XX);
for i = 1 : num_Views
	X{i} = XX{i}';
end



p_set = [1:-0.1:0.1];
lambda_set = 10.^[6:-1:-6];
if length(true_labels) < 5000
	m_set = 1;
	k_set = fix([1:-0.1:0.1] * length(true_labels));
else
	% k_set = [5:-1:1] * cluster_num;
	m_set = 1;
	k_set = [[3:3:15] * cluster_num, 1000];
end
% OrthNTF_Demos(datasetname, X, true_labels, cluster_num, m_set, k_set, p_set, lambda_set);  % have not to re-evaluate this method

alpha_set=10.^[-4:4];
gamma_set=10.^[-4:4];

p_set=[1:-0.1:0.1];
if length(true_labels) < 5000
	m_set= fix([1:-0.1:0.1]*length(true_labels))
else
	m_set= [[3:3:15] * cluster_num, 1000];
end


% TBGL_Demos2(datasetname, X, true_labels, cluster_num, m_set, alpha_set, gamma_set, p_set, seed)
end