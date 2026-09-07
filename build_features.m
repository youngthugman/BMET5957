% Build one 141-feature observation for every annotated second.
function [X, Y, patientID, featureNames] = build_features(dataPath, patientIndices)
data = matfile(dataPath); % Avoid loading the complete data set into memory.
ecgSize = size(data, 'ECG');
nPatients = ecgSize(2);

if nargin < 2 || isempty(patientIndices)
    patientIndices = 1:nPatients;
end

Fs = data.SR_ECG;
FsSpO2 = data.SR_SpO2;
featuresByPatient = cell(numel(patientIndices), 1);
labelsByPatient = cell(numel(patientIndices), 1);
idByPatient = cell(numel(patientIndices), 1);
featureNames = [];

for k = 1:numel(patientIndices)
    patient = patientIndices(k);
    fprintf("Processing patient %d...\n", patient);

    temp = data.ECG(1, patient);
    ecg = temp{1};
    temp = data.QRS(1, patient);
    qrs = temp{1};
    temp = data.SpO2(1, patient);
    spo2 = temp{1};
    temp = data.Class(1, patient);
    labels = char(temp{1});
    labels = labels(:);
    nSeconds = numel(labels);

    [ECGFeatures, ecgNames] = ...
        ECG_Feature_Extraction_Adjusted(ecg, qrs, Fs, nSeconds);
    [SpO2Features, spo2Names] = ...
        extract_spo2_features(spo2, FsSpO2, nSeconds);

    assert(size(ECGFeatures,1) == nSeconds, ...
        'ECG extractor must return one row per annotation second.');
    assert(size(ECGFeatures,2) == 125, ...
        'ECG extractor must return 125 features.');
    assert(size(SpO2Features,1) == nSeconds, ...
        'SpO2 extractor must return one row per annotation second.');
    assert(size(SpO2Features,2) == 16, ...
        'SpO2 extractor must return 16 features.');

    patientFeatures = [ECGFeatures SpO2Features];
    patientLabels = labels; % Preserve the original 1 Hz annotations unchanged.
    patientIDs = repmat(patient, nSeconds, 1);

    assert(size(patientFeatures,1) == nSeconds);
    assert(numel(patientLabels) == nSeconds);
    assert(numel(patientIDs) == nSeconds);

    featuresByPatient{k} = patientFeatures;
    labelsByPatient{k} = patientLabels;
    idByPatient{k} = patientIDs;

    if isempty(featureNames)
        featureNames = [string(ecgNames(:)).', string(spo2Names(:)).'];
    end
end

X = vertcat(featuresByPatient{:});
Y = vertcat(labelsByPatient{:});
patientID = vertcat(idByPatient{:});

assert(size(X,1) == numel(Y), 'X and Y are not aligned.');
assert(size(X,1) == numel(patientID), 'X and patientID are not aligned.');
assert(size(X,2) == numel(featureNames), ...
    'Feature data and feature names are not aligned.');
assert(size(X,2) == 141, 'Expected 125 ECG + 16 SpO2 features.');

fprintf('Feature extraction complete\n');
fprintf('Rows: %d\n', size(X,1));
fprintf('ECG features: 125\n');
fprintf('SpO2 features: 16\n');
fprintf('Total features: 141\n');
fprintf('A seconds: %d\n', sum(Y == 'A'));
fprintf('N seconds: %d\n', sum(Y == 'N'));
end
