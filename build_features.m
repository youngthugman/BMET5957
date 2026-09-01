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
FsSpO2 = data.SR_SpO2;
% temp ram storage
featuresByPatient = cell(numel(patientIndices), 1);
labelsByPatient   = cell(numel(patientIndices), 1);
idByPatient       = cell(numel(patientIndices), 1);

featureNames = []; 
for k = 1:numel(patientIndices)
    
    patient = patientIndices(k);                
    fprintf("Processing patient %d...\n", patient);
    %Load ECG
    temp = data.ECG(1, patient);
    ecg = temp{1};
    % loading labels
    temp = data.Class(1, patient);             
    labels = char(temp{1});
    labels = labels(:);
    nSeconds = numel(labels);                   % find seconds in recording from labels
    %Sp02 loading
    temp = data.SpO2(1, patient);               
    spo2 = temp{1};
    
   
    [ECGFeatures, ecgNames] = extract_ecg_features(ecg, Fs, nSeconds); % call our ECG feature extractor now each second of patient has 10 features
    [SpO2Features, spo2Names] = extract_spo2_features(spo2, FsSpO2,nSeconds);

    patientFeatures = [ECGFeatures SpO2Features];       %combine features
    featuresByPatient{k} = patientFeatures;
    labelsByPatient{k} = labels;
    idByPatient{k} = repmat(patient,nSeconds,1);
    
    if isempty(featureNames)
        featureNames = [ecgNames, spo2Names];
    end

    fprintf("  ECG features:  %d x %d\n", size(ECGFeatures,1), size(ECGFeatures,2));
    fprintf("  SpO2 features: %d x %d\n", size(SpO2Features,1), size(SpO2Features,2));
    fprintf("  Combined:      %d x %d\n\n", size(patientFeatures,1), size(patientFeatures,2));
end

% each row represents 1 second and is labelled like
% patientI--Second--feature1--feature2--...--label (N/A)
% Seperate rows but they align perfectly.
X = vertcat(featuresByPatient{:}); % make vertical for ML
Y = vertcat(labelsByPatient{:});   % make labels vertical
patientID = vertcat(idByPatient{:});





    
    





