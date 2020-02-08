function demo_ECFG_1()
% Demo for the ECFG algorithm proposed in the following paper:
% Dong Huang, Jian-Huang Lai and Chang-Dong Wang. 
% Ensemble Clustering Using Factor Graph, Pattern Recognition, 2016, 50, pp.131-142.

dataset =  'semeion';
% dataset =  'MNIST';

Msize = 20;   % The number of base clusterings
iterNum = 100; % Run ECFG $iterNum$ times and then get the average score.
 
%% ks        iterNum x Msize             8000  double  
% The i-th row in ks determines the construction of the ensemble in the
% i-th run. All rows in ks are randomly selected.
ks = zeros(iterNum, Msize);
for i = 1:iterNum
    tmp = randperm(200);
    ks(i,:) = tmp(1:Msize);
end


%% Loading the pool data.
gt = [];
load(['_',dataset,'_base_pool.mat']);

tinyFragThres = (numel(gt)^(1/2))/2; 

%% Parameters  
initialBasePb = 0.9;
pair_prior = 0.5;

result = [];
for idx = 1:size(ks,1) 
    disp(strcat('Idx: ', num2str(idx)));

    %% Get the set of base clusterings, each column one clustering.
    selectedIdx = ks(idx,:)';
    baseCls = members(:,selectedIdx); 

    %% Get super-objects
    tic;
    [newBaseCls, soClsLabels] = convert_into_super_objects(baseCls, tinyFragThres);
    toc;
    % The sizes of super-objects
    mcSizes = zeros(max(soClsLabels(:,2)),1); 
    for i = 1:numel(mcSizes)
        mcSizes(i) = sum(sum(soClsLabels(:,2)==i));
    end

    %% Prior: p(x_ij=1 | theta) 
    Px = ones(size(newBaseCls,1),size(newBaseCls,1))*pair_prior;    
    % The (initial) reliability of base clusterings
    Pr = ones(size(baseCls,2),1) * initialBasePb; 
    
    %% Get the co-association matrix
    S = zeros(size(newBaseCls,1),size(newBaseCls,1)); 
    M=[];
    for i = 1:size(newBaseCls,2)
        M{i} = getOneMatrixEAC(newBaseCls(:,i));
        S = S+M{i};
    end
    S = S/size(newBaseCls,2);
    
    iterMax = 3;
    for itIdx = 1:iterMax
        %% E-Step
        W = zeros(size(M{1}));
        for i = 1:size(newBaseCls,2)
            tmp = (-1).^(M{i}).*log((1-Pr(i))./Pr(i));
            W = W+tmp;
        end
        W = W + log(Px./(1-Px)); 


        %% Optimization of the BLP problem.
        xTic = tic;
        X = optimizeBLP(W);
        xToc = toc(xTic);

        %% The M-step
        for i = 1:numel(Pr)
            tmp = mcSizes*mcSizes';
            n_all_links = sum(sum(tmp.*triu(ones(size(X)),1)));
            n_right_links = sum(sum((M{i}.*X+(1-M{i}).*(1-X)).*tmp.*triu(ones(size(X)),1)));
            n_inner_links = sum(sum(diag(tmp)));
            Pr(i) = (n_right_links+n_inner_links)/(n_all_links+n_inner_links);
        end 
        Pr(Pr>0.95)=0.95;
    end
    
    labels = getConnCompLabels(X, 0);
    [~,I]=sort(soClsLabels(:,1));
    finalLabels = labels(soClsLabels(I,2));

    nmi = NMImax(finalLabels,gt)

    if itIdx == iterMax
        result = [result; nmi]; 
    end

    save(['result_',dataset,'_',num2str(Msize),'.mat','.mat'], 'result','ks');
   

end
