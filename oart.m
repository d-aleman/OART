function [anomalyLikelihood,LTM_values] = oart(T,trainRows)
% OART   Performs OART one-class classification 
%   [anomalyLikelihood,LTM_value] = oart(T,trainRows) returns the
%   likelihood of each sample in dataset in matrix T being anomalous in 
%   anomalyLikelihood and the long-term memory (LTM) values found for each 
%   feature in LTM_values. Training is performed on dataset row indices in
%   vector trainRows.
%
%   Example:
%       [anomalyLikelihood,LTM_values] = oart(T,trainRows)
%
%   See also OART_TRAIN, OART_TEST.


%% Load and prepare the dataset

% Remove empty columns from dataset
T(:,sum(abs(T),1)==0) = [];

% Obtain training dataset
T_train = T(trainRows,:);

% Get size of datasets
[numSamples, numFeatures] = size(T);
numTrain = size(T_train,1);

% Linear transformation into [0,1]; comment out to switch scaling off. 
T = (T-repmat(min(T,[],1),numSamples,1)) ./ repmat((max(T,[],1)-min(T,[],1)),numSamples,1);


%% Constants

% OART parameters
distances = 0.01:0.02:0.99;    % range of distance values to measure against stored feature values
numStableFindingIters = 10;    % number of stable-feature-finding iterations to find significantly stable features
stableThreshold = 1;           % percentage of stable-finding iterations survived to be consider significant

numDistances = length(distances);

% PART training parameters
numRunsPART = 20;           % number of PART runs in training to find stable features
numShuffles = 2;            % number of full training datasets to concatenate for training
alpha = 0.1;                % learning rate for top-down weights (stored feature values)
stability = 0.9;            % percentage of runs that dimensions should survive to be stable significant
K = numShuffles * numTrain; % number of iterations to update weights


%% Iterate the training process for each testing sample that is left out

% Initiate the stable memory features and their long-term values
significantNA = zeros(numDistances,numStableFindingIters,numFeatures);  % features that are significant for non-anomalous samples
LTM_stored = zeros(numDistances,numStableFindingIters,numFeatures);     % long-term memory (LTM) stored values
stableSigFeatures = zeros(numDistances,numFeatures);                    % stably significant features per distance

% For each distance and stable-finding run, run OART algorithm
for dd = 1:numDistances
    
    distance = distances(dd);  

    for ii = 1:numStableFindingIters

        % Train and store mean LTM values, significant features just for
        % non-anomalies; trainDetails can be used for detailed analysis
        [mean_LTM,significantFeatures] = oart_train(T_train,distance,numRunsPART,alpha,stability,K);
        LTM_stored(dd,ii,:) = mean_LTM;
        significantNA(dd,ii,significantFeatures) = 1;
    end

    
    stableSigFeatures(dd,squeeze(sum(significantNA(dd,:,:),2) >= numStableFindingIters*stableThreshold)) = 1;

end

[deviation,LTM_values] = oart_test(T,LTM_stored,stableSigFeatures,distances);

% Likelihood of sample being anomalous is the sum of deviations over all
% features and distances
anomalyLikelihood = squeeze(nansum(nansum(deviation,2),1)); 

