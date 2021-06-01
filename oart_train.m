function [mean_LTM,significantFeatures] = oart_train(T_train,distance,numRunsPART,alpha,stability,K)
% OART_TRAIN  Trains the OART one-class classifier on dataset T_train.
%   [mean_LTM,significantFeatures] = oart_train(T_train,trainParams,distance)
%   trains using the specified distance and PART parameters, and returns
%   the mean long-term memory (LTM) values in mean_LTM and the indices of
%   significant features in significantFeatures. The training parameters
%   are the number of PART runs (numRunsPART), th learning rate for
%   top-down weights (alpha), the percentage of survived dimensions to be
%   stably significant (stability), and the number of iterations to update
%   weights (K).
% 
% Example: 
%   [mean_LTM,significantFeatures] = oart_train(T_train,trainParams,distance);
%
% See also: OART, OART_TEST.


%% Constants

[numTrain,numFeatures] = size(T_train);

% Training set values; default is matrix of ones
M = ones(size(T_train));


%% Perform PART numRunsPART times

survivedFeatures = zeros(numFeatures,numRunsPART);  % dimensions with selective signal=1 after all K training
LTM_temp = zeros(numFeatures,numRunsPART);          % temporary LTM values

% For each PART run
for rr = 1:numRunsPART

    h = zeros(numFeatures,K+1);
    hsigma = zeros(numFeatures,K);
    
    % Bottom-up weights; +1 size to calculate the stored feature values for later based on the last input
    l = zeros(numFeatures,K+1);
    Z = zeros(numFeatures,K+1);

    % Permuted order of samples to use in this run
    permarray = cat(2,randperm(numTrain),randperm(numTrain));

    % Initial top-down class, using just first permuted sample
    Z(:,1) = T_train(permarray(1),:)';

    % Initial weights
    l(:,1) = 1;
    h(:,1) = 1;
    
    % For each training run within PART
    for k = 1:K

        % Find and store features within specified distance
        isWithinDistance = abs(T_train(permarray(k),:)' - Z(:,k)) <= distance;
        hsigma(isWithinDistance, k) = 1;
        h(:,k) = hsigma(:,k) .* l(:,k);
        
        % Update top-down weights
        val = T_train(permarray(k),:) .* M(permarray(k),:);
        randSampleOrder = Z(:,k) .* (1 - (M(permarray(k),:))');
        Z(:,k+1) = ((1-alpha) * Z(:,k)) + alpha * (val)' + (randSampleOrder);

        % Update bottom-up weights according to found features
        l(h(:,k)==1,k+1) = 1;
        l(h(:,k)==0,k+1) = 0;

    end
   
    % Store temporary LTM values and features that survived at end of
    % PART training
    LTM_temp(:,rr) = Z(:,end);      
    survivedFeatures(l(:,end)==1,rr) = 1;
   
end

% LTM values found in training are the mean of LTM values
mean_LTM = mean(LTM_temp,2);

% Significant features are those that survived specified percent of PART runs
significantFeatures = find(sum(survivedFeatures(:,:),2) >= stability*numRunsPART);
