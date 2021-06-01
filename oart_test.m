function [deviation,LTM_values] = oart_test(T,LTM_stored,stableSigFeatures,distances)
% OART_TEST    Test OART one-class classification performance.
%   [deviation,LTM_values] = oart_test(T,LTM_stored,stableSigFeatures,distances)
%   tests each sample in dataset T by determining the deviation in feature
%   values from stored long-term memory (LTM) values (LTM_stored) in the
%   stably significant features (stableSigFeatures) for each distance in
%   array distances.
%
%   Example:
%       [deviation,LTM_values] = oart_test(T,LTM_stored,stableSigFeatures,distances)
%
%   See also: OART, OART_TRAIN.


%% Constants

[numSamples,numFeatures] = size(T);


%% Compute deviation values

deviation = NaN(length(distances),numFeatures,numSamples);
LTM_values = zeros(length(distances),numFeatures);

for distance = 1:length(distances)
    
    stableFeaturesThisDistance = find(stableSigFeatures(distance,:) == 1);
    
    % Final long-term memory values are features that are within specified
    % distance of stored LTM values
    LTM_distance = squeeze(mean(LTM_stored(distance,:,:),2))';
    LTM_values(distance,:) = LTM_distance;
    
    % Deviation from stored LTM of significant features
    deviation(distance,stableFeaturesThisDistance,:) = abs(T(:,stableFeaturesThisDistance)-repmat(LTM_distance(stableFeaturesThisDistance),numSamples,1))';
end

