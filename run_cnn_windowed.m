%% Windowed raw ECG + SpO2 1D CNN development experiment
% One centred 11-second signal window produces one label for its centre second.

clear
clc

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot,"Classifier"));

dataPath = "E:\Desktop\Downloads\ProjectTrainData.mat";

ecgRate = 50;
contextSeconds = 11;
halfContext = 5;
threshold = 0.50;

expectedPatients = ...
    [1 3 9 12 15 17 18 30 33 36 37 38 42 47 50 57 61 72 77 94];

rng(1);
patients = sort(randperm(100,20));

assert(isequal(patients,expectedPatients), ...
    "The deterministic development patient subset has changed.");

fprintf("Windowed raw ECG + SpO2 1D CNN\n");
fprintf("Patients: %d\n",numel(patients));
fprintf("Selected patient IDs: %s\n",mat2str(patients));
fprintf("ECG rate: %d Hz\n",ecgRate);
fprintf("ECG context: %d s\n",contextSeconds);
fprintf("ECG samples/window: %d\n",contextSeconds * ecgRate);
fprintf("SpO2 rate: 1 Hz\n");
fprintf("SpO2 context: %d s\n",contextSeconds);
fprintf("SpO2 samples/window: %d\n",contextSeconds);
fprintf("Window stride: 1 s\n");
fprintf("Predictions/window: 1\n");
fprintf("Output interval: 1 s\n\n");

S = load( ...
    dataPath, ...
    "ECG", ...
    "SpO2", ...
    "Class", ...
    "SR_ECG", ...
    "SR_SpO2");

assert(max(patients) <= numel(S.Class), ...
    "The data set has fewer than 100 patients.");

%% Read labels first

patientLabels = cell(numel(patients),1);
patientSeconds = zeros(numel(patients),1);

for i = 1:numel(patients)

    p = patients(i);

    patientLabels{i} = cleanLabels(S.Class{p});
    patientSeconds(i) = numel(patientLabels{i});

    assert(patientSeconds(i) > 0, ...
        "A selected patient has no annotation seconds.");

end

totalSeconds = sum(patientSeconds);

%% Allocate fixed-size raw windows

XECG = zeros( ...
    contextSeconds * ecgRate, ...
    1, ...
    totalSeconds, ...
    "single");

XSpO2 = zeros( ...
    contextSeconds, ...
    1, ...
    totalSeconds, ...
    "single");

labelText = strings(totalSeconds,1);
patientID = zeros(totalSeconds,1);

patientWindowIndices = cell(numel(patients),1);

nextWindow = 1;

%% Construct one centred window for every second

for i = 1:numel(patients)

    p = patients(i);

    labels = patientLabels{i};
    nSeconds = numel(labels);

    ecg = prepareECG( ...
        S.ECG{p}, ...
        scalarRate(S.SR_ECG,p), ...
        ecgRate, ...
        nSeconds);

    spo2 = prepareSpO2( ...
        S.SpO2{p}, ...
        scalarRate(S.SR_SpO2,p), ...
        nSeconds);

    % Split ECG into individual 1-second columns.
    ecgBySecond = reshape(ecg,ecgRate,nSeconds);

    % Pad five seconds before and after.
    % Only signals are padded. Labels are untouched.
    paddedECG = [ ...
        repmat(ecgBySecond(:,1),1,halfContext), ...
        ecgBySecond, ...
        repmat(ecgBySecond(:,end),1,halfContext)];

    paddedECG = paddedECG(:);

    paddedSpO2 = [ ...
        repmat(spo2(1),halfContext,1); ...
        spo2; ...
        repmat(spo2(end),halfContext,1)];

    indices = ...
        nextWindow:(nextWindow+nSeconds-1);

    patientWindowIndices{i} = indices(:);

    for t = 1:nSeconds

        destination = indices(t);

        firstSample = ...
            (t-1)*ecgRate + 1;

        XECG(:,1,destination) = ...
            paddedECG( ...
                firstSample: ...
                firstSample + contextSeconds*ecgRate - 1);

        XSpO2(:,1,destination) = ...
            paddedSpO2( ...
                t: ...
                t + contextSeconds - 1);

    end

    labelText(indices) = string(labels);
    patientID(indices) = p;

    assert(numel(indices) == numel(S.Class{p}), ...
        "Windows for patient %d do not equal annotation seconds.",p);

    nextWindow = nextWindow + nSeconds;

end

%% Final categorical targets

Y = categorical(labelText,{'N','A'});

assert(~any(isundefined(Y)), ...
    "Class contains a label other than N or A.");

assert(isequal(string(categories(Y)),["N";"A"]), ...
    "Class order must be N followed by A.");

assert(nextWindow-1 == totalSeconds, ...
    "Window count does not equal annotation count.");

assert(size(XECG,3) == totalSeconds, ...
    "ECG window count is incorrect.");

assert(size(XSpO2,3) == totalSeconds, ...
    "SpO2 window count is incorrect.");

%% Dataset summary

fprintf("Total labelled seconds: %d\n",totalSeconds);
fprintf("Total windows: %d\n",size(XECG,3));
fprintf("A seconds: %d\n",sum(Y == "A"));
fprintf("N seconds: %d\n\n",sum(Y == "N"));

windowBytes = ...
    4 * (numel(XECG) + numel(XSpO2));

fprintf( ...
    "Window-array RAM (XECG + XSpO2): %.2f GiB\n\n", ...
    windowBytes / 2^30);

%% Network sanity check

sanityNet = train_cnn_windowed();

sanityECG = ...
    dlarray(zeros(1,1,550,"single"),"CBT");

sanitySpO2 = ...
    dlarray(zeros(1,1,11,"single"),"CBT");

sanityScores = predict( ...
    sanityNet, ...
    sanityECG, ...
    sanitySpO2);

assert( ...
    numel(sanityScores) == 2 && ...
    size(sanityScores,1) == 2, ...
    "One observation must produce exactly two scores [N; A].");

fprintf("CNN sanity check passed:\n");
fprintf("  ECG input:  550 samples\n");
fprintf("  SpO2 input: 11 samples\n");
fprintf("  Output:      2 class scores\n\n");

clear sanityNet sanityScores sanityECG sanitySpO2

%% 5-fold patient-wise cross validation

rng(1);
cv = cvpartition(numel(patients),"KFold",5);

metrics = zeros(5,4);

for fold = 1:5

    trainPatients = ...
        patients(training(cv,fold));

    valPatients = ...
        patients(test(cv,fold));

    assert( ...
        isempty(intersect(trainPatients,valPatients)), ...
        "Patient leakage detected.");

    trainWindows = ...
        ismember(patientID,trainPatients);

    fprintf("\nFold %d\n",fold);
    fprintf("Training patients: %s\n",mat2str(trainPatients));
    fprintf("Validation patients: %s\n",mat2str(valPatients));
    fprintf("Training windows: %d\n",sum(trainWindows));

    % All overlapping windows from a patient stay in that patient's fold.
    net = train_cnn_windowed( ...
        XECG(:,:,trainWindows), ...
        XSpO2(:,:,trainWindows), ...
        Y(trainWindows));

    truth = ...
        categorical(strings(0,1),{'N','A'});

    predictedA = false(0,1);

%% Validation patient by patient

valPatientIndices = find(ismember(patients,valPatients));

for k = 1:numel(valPatientIndices)

    i = valPatientIndices(k);

    idx = patientWindowIndices{i};

    probabilityA = predictProbabilityA( ...
        net, ...
        XECG(:,:,idx), ...
        XSpO2(:,:,idx), ...
        512);

    expectedSeconds = patientSeconds(i);

    assert( ...
        numel(probabilityA) == expectedSeconds, ...
        "Validation predictions do not equal annotation seconds.");

    assert( ...
        numel(idx) == expectedSeconds, ...
        "Validation window count does not equal annotation seconds.");

    predictedA = [ ...
        predictedA; ...
        probabilityA >= threshold]; %#ok<AGROW>

    patientTruth = patientLabels{i};

    assert( ...
        numel(patientTruth) == expectedSeconds, ...
        "Validation truth length does not equal annotation seconds.");

    truth = [ ...
        truth; ...
        patientTruth]; %#ok<AGROW>

end

    %% Metrics

    actualA = truth == "A";

    TP = sum(predictedA & actualA);
    FP = sum(predictedA & ~actualA);
    FN = sum(~predictedA & actualA);
    TN = sum(~predictedA & ~actualA);

    sensitivity = safeDivide(TP,TP+FN);
    ppv = safeDivide(TP,TP+FP);
    f1 = safeDivide(2*TP,2*TP+FP+FN);
    accuracy = safeDivide(TP+TN,TP+FP+FN+TN);

    metrics(fold,:) = ...
        [sensitivity ppv f1 accuracy];

    fprintf( ...
        "Fold %d: Sens = %.3f, PPV = %.3f, F1 = %.3f, Accuracy = %.3f\n", ...
        fold, ...
        sensitivity, ...
        ppv, ...
        f1, ...
        accuracy);

end

%% Final CV summary

fprintf("\n5-Fold Patient-Wise Cross Validation\n");
fprintf("Mean Sensitivity: %.3f\n",mean(metrics(:,1)));
fprintf("Mean PPV: %.3f\n",mean(metrics(:,2)));
fprintf("Mean F1: %.3f\n",mean(metrics(:,3)));
fprintf("Mean Accuracy: %.3f\n",mean(metrics(:,4)));
fprintf("Std F1: %.3f\n",std(metrics(:,3)));

%% Local functions

function labels = cleanLabels(raw)

labels = categorical( ...
    upper(strtrim(string(raw(:)))), ...
    {'N','A'});

assert(~any(isundefined(labels)), ...
    "Class contains a label other than N or A.");

end

function fs = scalarRate(rates,p)

if iscell(rates)

    fs = rates{p};

elseif isscalar(rates)

    fs = rates;

else

    fs = rates(p);

end

fs = double(fs(1));

assert(isfinite(fs) && fs > 0, ...
    "Invalid sampling rate.");

end

function ecg = prepareECG(raw,fs,targetFs,nSeconds)

ecg = double(raw(:));

ecg(~isfinite(ecg)) = NaN;

ecg = fillmissing( ...
    ecg, ...
    "linear", ...
    EndValues="nearest");

assert(~isempty(ecg) && ~any(isnan(ecg)), ...
    "ECG contains no usable samples.");

% Anti-aliased polyphase resampling.
[p,q] = rat(targetFs/fs,1e-12);
ecg = resample(ecg,p,q);

needed = nSeconds * targetFs;

if numel(ecg) < needed
    ecg(end+1:needed) = ecg(end);
end

ecg = ecg(1:needed);

% Robust per-record normalization.
centre = median(ecg);

scale = ...
    1.4826 * median(abs(ecg-centre));

if ~isfinite(scale) || scale < eps
    scale = std(ecg);
end

if ~isfinite(scale) || scale < eps
    scale = 1;
end

ecg = (ecg-centre) / scale;

% Limit extreme artefacts.
ecg = max(-10,min(10,ecg));

ecg = single(ecg(:));

end

function spo2 = prepareSpO2(raw,fs,nSeconds)

spo2 = double(raw(:));

spo2(spo2 == 0 | ~isfinite(spo2)) = NaN;

spo2 = fillmissing( ...
    spo2, ...
    "linear", ...
    EndValues="nearest");

assert(~isempty(spo2) && ~any(isnan(spo2)), ...
    "SpO2 contains no usable samples.");

% Align to one value per annotation second.
if abs(fs-1) > 1e-6

    sourceTime = ...
        (0:numel(spo2)-1)' / fs;

    targetTime = ...
        (0:nSeconds-1)';

    spo2 = interp1( ...
        sourceTime, ...
        spo2, ...
        targetTime, ...
        "linear", ...
        "extrap");

end

if numel(spo2) < nSeconds
    spo2(end+1:nSeconds) = spo2(end);
end

spo2 = spo2(1:nSeconds);

% Fixed physiological scaling.
spo2 = (spo2-90) / 10;
spo2 = max(-2,min(2,spo2));

spo2 = single(spo2(:));

end

function probabilityA = predictProbabilityA( ...
    net,ecg,spo2,batchSize)

n = size(ecg,3);

probabilityA = zeros(n,1,"single");

for first = 1:batchSize:n

    idx = ...
        first:min(first+batchSize-1,n);

    % Stored data:
    % time x channel x batch
    %
    % dlnetwork:
    % channel x time x batch
    ecgBatch = dlarray( ...
        permute(ecg(:,:,idx),[2 1 3]), ...
        "CTB");

    spo2Batch = dlarray( ...
        permute(spo2(:,:,idx),[2 1 3]), ...
        "CTB");

    scores = predict( ...
        net, ...
        ecgBatch, ...
        spo2Batch);

    scores = ...
        gather(extractdata(scores));

    scores = reshape(scores,2,[]);

    assert(size(scores,2) == numel(idx), ...
        "Expected one score pair per window.");

    probabilityA(idx) = ...
        scores(2,:).';

end

end

function value = safeDivide(numerator,denominator)

if denominator == 0
    value = 0;
else
    value = numerator / denominator;
end

end