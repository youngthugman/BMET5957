%% This section 


% Inputs are the data path and which patients we want to process e.g. 1:50.
% Output is X = ECG and SpO2 features for each 60-second epoch, Y = matching
% N/A labels, patient ID and names of features.
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
    % Load the supplied approximate QRS detections used by the ECG extractor.
    temp = data.QRS(1, patient);
    qrs = temp{1};
    % loading labels
    temp = data.Class(1, patient);             
    labels = char(temp{1});
    labels = labels(:);
    nSeconds = numel(labels);                   % find seconds in recording from labels
    %Sp02 loading
    temp = data.SpO2(1, patient);               
    spo2 = temp{1};
    
    [ECGFeatures, epochLabels, ~, epochTimes, ecgNames] = ...
        ECG_Feature_Extraction_Adjusted({ecg}, {qrs}, {labels}, Fs);

    % The ECG extractor emits one row per 60-second epoch. The SpO2
    % extractor remains at 1 Hz, so retain the row at each epoch centre to
    % align its centred five-minute window with the corresponding ECG row.
    [SpO2FeaturesPerSecond, spo2Names] = ...
        extract_spo2_features(spo2, FsSpO2, nSeconds);
    spo2Rows = round(epochTimes + 0.5);
    if any(spo2Rows < 1 | spo2Rows > nSeconds)
        error("ECG epoch times do not align with the SpO2 feature rows.");
    end
    SpO2Features = SpO2FeaturesPerSecond(spo2Rows,:);

    patientFeatures = [ECGFeatures SpO2Features];       %combine features
    featuresByPatient{k} = patientFeatures;
    labelsByPatient{k} = char(epochLabels);
    idByPatient{k} = repmat(patient,size(patientFeatures,1),1);
    
    if isempty(featureNames)
        featureNames = [string(ecgNames(:)).', string(spo2Names(:)).'];
    end

    fprintf("  ECG features:  %d x %d\n", size(ECGFeatures,1), size(ECGFeatures,2));
    fprintf("  SpO2 features: %d x %d\n", size(SpO2Features,1), size(SpO2Features,2));
    fprintf("  Combined:      %d x %d\n\n", size(patientFeatures,1), size(patientFeatures,2));
end

% each row represents one 60-second epoch and is labelled like
% patientI--Epoch--feature1--feature2--...--label (N/A)
% Seperate rows but they align perfectly.
X = vertcat(featuresByPatient{:}); % make vertical for ML
Y = vertcat(labelsByPatient{:});   % make labels vertical
patientID = vertcat(idByPatient{:});





    
    




