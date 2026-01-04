function MultiEvalutions(datasetname, XX, true_labels,cluster_num,init_labels)


layer_set=[10*cluster_num,5*cluster_num,1*cluster_num;15*cluster_num,5*cluster_num,1*cluster_num;15*cluster_num,10*cluster_num,1*cluster_num;];
lambda1_set= 10.^[-2:1];
lambda2_set= 10.^[-2:1];
lambda3_set= 10.^[-2:1];
p_set=[0.1:0.1:1];

TNAGL_Demos(datasetname,XX,true_labels,cluster_num,layer_set,lambda1_set,lambda2_set,lambda3_set,p_set);


end
