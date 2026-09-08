%% Raw ECG + SpO2 sequence-to-sequence CNN experiment
% A 60-second input produces 60 independent, second-level predictions.
clear; clc;

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot,'Classifier'));

dataPath = 'E:\Desktop\Downloads\ProjectTrainData.mat';
cnnRate = 50;
blockSeconds = 60;
threshold = 0.50;

rng(1);
patients = sort(randperm(100,20));
fprintf('Raw ECG + SpO2 1D CNN\n');
fprintf('Patients: 20\nSelected patient IDs: %s\n', mat2str(patients));
fprintf('ECG CNN rate: %d Hz\n', cnnRate);
fprintf('Input block length: %d s\n', blockSeconds);
fprintf('Full block samples: %d\n', blockSeconds * cnnRate);
fprintf('Outputs per full block: %d\n', blockSeconds);
fprintf('Output interval: 1 s\n\n');

S = load(dataPath, 'ECG', 'SpO2', 'Class', 'SR_ECG', 'SR_SpO2');
assert(max(patients) <= numel(S.Class), 'The data set has fewer than 100 patients.');

X = {}; Y = {}; blockPatientID = []; blockNumber = [];
patientLabels = cell(numel(patients),1);
patientBlockIndices = cell(numel(patients),1);
for i = 1:numel(patients)
    p = patients(i);
    labels = cleanLabels(S.Class{p});
    nSeconds = numel(labels);
    patientLabels{i} = labels;

    % Signal preparation is deliberately label-independent. ECG uses a
    % per-record robust scale; SpO2 uses fixed (value - 90)/10 scaling.
    ecg = prepareECG(S.ECG{p}, scalarRate(S.SR_ECG,p), cnnRate, nSeconds);
    spo2 = prepareSpO2(S.SpO2{p}, scalarRate(S.SR_SpO2,p), nSeconds);
    signal = [ecg; repelem(spo2, cnnRate)];
    assert(size(signal,2) == nSeconds * cnnRate);

    firstBlock = numel(X) + 1;
    for firstSecond = 1:blockSeconds:nSeconds
        lastSecond = min(firstSecond + blockSeconds - 1, nSeconds);
        sampleRange = (firstSecond-1)*cnnRate + (1:(lastSecond-firstSecond+1)*cnnRate);
        X{end+1,1} = single(signal(:,sampleRange)); %#ok<SAGROW>
        Y{end+1,1} = labels(firstSecond:lastSecond).'; %#ok<SAGROW>
        blockPatientID(end+1,1) = p; %#ok<SAGROW>
        blockNumber(end+1,1) = numel(X) - firstBlock + 1; %#ok<SAGROW>
        assert(size(X{end},2) / cnnRate == numel(Y{end}));
    end
    patientBlockIndices{i} = (firstBlock:numel(X)).';
    assert(sum(cellfun(@numel,Y(patientBlockIndices{i}))) == nSeconds);
end

allLabels = vertcat(patientLabels{:});
totalTargets = sum(cellfun(@numel,Y));
fprintf('Total labelled seconds: %d\n', numel(allLabels));
fprintf('Total CNN output targets: %d\n', totalTargets);
fprintf('Total blocks: %d\n', numel(X));
fprintf('Number of A seconds: %d\n', sum(allLabels == 'A'));
fprintf('Number of N seconds: %d\n\n', sum(allLabels == 'N'));
assert(totalTargets == numel(allLabels), 'A labelled second was lost.');

% Partition patient indices, never blocks or seconds.
rng(1);
cv = cvpartition(numel(patients), 'KFold', 5);
metrics = zeros(5,4);
for fold = 1:5
    trainPatientMask = training(cv,fold);
    valPatientMask = test(cv,fold);
    trainPatients = patients(trainPatientMask);
    valPatients = patients(valPatientMask);
    assert(isempty(intersect(trainPatients,valPatients)), 'Patient leakage detected.');
    trainBlockMask = ismember(blockPatientID,trainPatients);

    % train_cnn derives weights exclusively from these training targets.
    net = train_cnn(X(trainBlockMask), Y(trainBlockMask));

    truth = categorical(strings(0,1), {'N','A'});
    predictedA = false(0,1);
    for i = find(valPatientMask).'
        idx = patientBlockIndices{i};
        [~,order] = sort(blockNumber(idx));
        idx = idx(order);
        patientProbabilityA = zeros(0,1);
        for b = idx.'
            scores = predict(net, dlarray(X{b},'CT'));
            scores = gather(extractdata(scores));
            assert(size(scores,1) == 2 && size(scores,2) == numel(Y{b}), ...
                'CNN output is not one score pair per annotation second.');
            patientProbabilityA = [patientProbabilityA; scores(2,:).']; %#ok<AGROW>
        end
        assert(numel(patientProbabilityA) == numel(patientLabels{i}), ...
            'Predicted seconds do not equal annotation seconds.');
        predictedA = [predictedA; patientProbabilityA >= threshold]; %#ok<AGROW>
        truth = [truth; patientLabels{i}]; %#ok<AGROW>
    end

    actualA = truth == 'A';
    TP = sum(predictedA & actualA); FP = sum(predictedA & ~actualA);
    FN = sum(~predictedA & actualA); TN = sum(~predictedA & ~actualA);
    sensitivity = TP / max(TP+FN,1);
    ppv = TP / max(TP+FP,1);
    f1 = 2*TP / max(2*TP+FP+FN,1);
    accuracy = (TP+TN) / (TP+FP+FN+TN);
    metrics(fold,:) = [sensitivity ppv f1 accuracy];
    fprintf('Fold %d: Sens = %.3f, PPV = %.3f, F1 = %.3f, Accuracy = %.3f\n', ...
        fold, sensitivity, ppv, f1, accuracy);
end
fprintf('\nMean Sensitivity: %.3f\n', mean(metrics(:,1)));
fprintf('Mean PPV: %.3f\n', mean(metrics(:,2)));
fprintf('Mean F1: %.3f\n', mean(metrics(:,3)));
fprintf('Mean Accuracy: %.3f\n', mean(metrics(:,4)));
fprintf('Std F1: %.3f\n', std(metrics(:,3)));

function labels = cleanLabels(raw)
labels = categorical(upper(strtrim(string(raw(:)))), {'N','A'});
assert(~any(isundefined(labels)), 'Class contains a label other than N or A.');
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
assert(isfinite(fs) && fs > 0, 'Invalid sampling rate.');
end

function ecg = prepareECG(raw,fs,targetFs,nSeconds)
ecg = double(raw(:));
ecg(~isfinite(ecg)) = NaN;
ecg = fillmissing(ecg,'linear','EndValues','nearest');
assert(~any(isnan(ecg)), 'ECG contains no usable samples.');
[p,q] = rat(targetFs/fs,1e-12);
ecg = resample(ecg,p,q); % polyphase anti-aliasing filter
needed = nSeconds * targetFs;
if numel(ecg) < needed, ecg(end+1:needed) = ecg(end); end
ecg = ecg(1:needed);
centre = median(ecg);
scale = 1.4826 * median(abs(ecg-centre));
if ~isfinite(scale) || scale < eps, scale = std(ecg); end
if ~isfinite(scale) || scale < eps, scale = 1; end
ecg = max(-10,min(10,(ecg-centre)/scale)).';
end

function spo2 = prepareSpO2(raw,fs,nSeconds)
spo2 = double(raw(:));
spo2(spo2 == 0 | ~isfinite(spo2)) = NaN;
spo2 = fillmissing(spo2,'linear','EndValues','nearest');
assert(~any(isnan(spo2)), 'SpO2 contains no usable samples.');
% Sample the filled record at annotation-second centres if its stated rate
% differs from 1 Hz, then pad/truncate small acquisition-length mismatches.
if abs(fs-1) > 1e-6
    oldTime = (0:numel(spo2)-1)'/fs;
    spo2 = interp1(oldTime,spo2,(0:nSeconds-1)','linear','extrap');
end
if numel(spo2) < nSeconds, spo2(end+1:nSeconds) = spo2(end); end
spo2 = spo2(1:nSeconds);
spo2 = max(-2,min(2,(spo2-90)/10)).';
end
