function [features, featureNames, qrs] = extract_ecg_features(ecg, Fs, nSeconds, options)
%EXTRACT_ECG_FEATURES Main entry point for the Austin ECG module.
%
% INPUT
%   ecg       - raw ECG vector
%   Fs        - ECG sampling frequency, normally 200 Hz
%   nSeconds  - number of annotated seconds
%
% OUTPUT
%   features      - nSeconds x 10 feature matrix
%   featureNames  - 1 x 10 string array
%   qrs           - detected QRS sample indices

arguments
    ecg
    Fs (1,1) double
    nSeconds (1,1) double

    options.contextBefore (1,1) double = 20
    options.contextAfter  (1,1) double = 20
    options.minRR         (1,1) double = 0.30
    options.maxRR         (1,1) double = 2.00
    options.qrsMode       (1,1) string = "existing"
    options.suppliedQRS = []
end

% Minimal configuration required by existing ECG adapter
config.contextBeforeSeconds = options.contextBefore;
config.contextAfterSeconds  = options.contextAfter;

config.ecg.qrsMode = options.qrsMode;
config.ecg.minRR   = options.minRR;
config.ecg.maxRR   = options.maxRR;

% Run existing Austin ECG module
[features, featureNames, qrs] = ...
    extract_ecg_features_adapter( ...
        ecg, ...
        Fs, ...
        nSeconds, ...
        config, ...
        options.suppliedQRS);

% Standardised output
assert(size(features,1) == nSeconds, ...
    "ECG extractor must return one row per second.");

assert(size(features,2) == numel(featureNames), ...
    "Feature matrix does not match feature names.");

qrs = qrs(:);

end