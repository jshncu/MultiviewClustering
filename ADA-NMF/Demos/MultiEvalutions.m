function MultiEvalutions(datasetname, XX, true_labels,cluster_num)



%%%%%%%%%%%%%%%%%%%%%%%%%%%DANMF%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
layer_set=[10*cluster_num,5*cluster_num,1*cluster_num;15*cluster_num,5*cluster_num,1*cluster_num;20*cluster_num,5*cluster_num,1*cluster_num;25*cluster_num,5*cluster_num,1*cluster_num;
    15*cluster_num,10*cluster_num,1*cluster_num;20*cluster_num,10*cluster_num,1*cluster_num;25*cluster_num,10*cluster_num,1*cluster_num;
    20*cluster_num,15*cluster_num,1*cluster_num;25*cluster_num,15*cluster_num,1*cluster_num;25*cluster_num,20*cluster_num,1*cluster_num;];

lambda_set= 10.^[-3:3];
alpha_set=10.^[-3:3];
beta_set=10.^[-3:3];
p_set=[1];
DANMF_Demos(datasetname,XX,true_labels,cluster_num,layer_set,lambda_set,alpha_set,beta_set,p_set);
end
