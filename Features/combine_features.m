function [X,names] = combine_features(ecgFeatures,ecgNames,spo2Features,spo2Names,annotationLength)
%COMBINE_FEATURES Enforce exact per-second alignment before concatenation.
assert(size(ecgFeatures,1)==size(spo2Features,1),'ECG and SpO2 feature row counts differ.');
assert(size(ecgFeatures,1)==annotationLength,'Feature rows do not equal annotation length.');
X=[ecgFeatures spo2Features]; names=[string(ecgNames) string(spo2Names)];
assert(size(X,2)==numel(names),'Feature-name count does not match feature columns.');
end
