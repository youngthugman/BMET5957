%% This section 


% Inputs are the data path and which patients we want to process e.g. 1:50
% or something. Output is X = ecg features for each second of the patient,
% Y = Matching N/A labels, patient ID and names of features.
function [X, Y, patientID, featureNames] = build_features(dataPath, patientIndices)
data = matfile(dataPath); % do not use load() as its like 1.2 gb
ecgSize = size(data, 'ECG'); 
nPatients = ecgSize(2);

if nargin < 2 || isempty(patientIndices)       % fall back if no patients selected
    patientIndices = 1:nPatients;              % select all patients
end

Fs = data.SR_ECG;                               % Sampling Rate
% temp ram storage
featuresByPatient = cell(numel(patientIndices), 1);
labelsByPatient   = cell(numel(patientIndices), 1);
idByPatient       = cell(numel(patientIndices), 1);

featureNames = []; 
for k = 1:numel(patientIndices)
    patient = patientIndices(k);                % loading ECG
    temp = data.ECG(1, patient);
    ecg = temp{1};
    temp = data.Class(1, patient);              % loading labels
    labels = char(temp{1});
    labels = labels(:);
    nSeconds = numel(labels);                   % find seconds in recording from labels
    [ECGFeatures, names] = extract_ecg_features(ecg, Fs, nSeconds); % call our ECG feature extractor now each second of patient has 10 features
    featuresByPatient{k} = ECGFeatures;
    labelsByPatient{k} = labels;
    idByPatient{k} = repmat(patient,nSeconds,1);
    if isempty(featureNames)
        featureNames = names;
    end
end

% each row represents 1 second and is labelled like
% patientI--Second--feature1--feature2--...--label (N/A)
% Seperate rows but they align perfectly.
X = vertcat(featuresByPatient{:}); % make vertical for ML
Y = vertcat(labelsByPatient{:});   % make labels vertical
patientID = vertcat(idByPatient{:});





    
    





