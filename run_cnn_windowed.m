%% Windowed raw ECG + SpO2 1D CNN development experiment
% One centred 11-second signal window produces one label for its centre second.
clear; clc;

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot,"Classifier"));

dataPath = "C:\Users\maxas\Downloads\ProjectTrainData_ML.mat";
ecgRate = 50;
contextSeconds = 11;
halfContext = 5;
threshold = 0.50;
expectedPatients = [1 3 9 12 15 17 18 30 33 36 37 38 42 47 50 57 61 72 77 94];

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

S = load(dataPath,"ECG","SpO2","Class","SR_ECG","SR_SpO2");
assert(max(patients) <= numel(S.Class),"The data set has fewer than 100 patients.");

% Count targets before allocating the fixed-size window arrays.
patientLabels = cell(numel(patients),1);
patientSeconds = zeros(numel(patients),1);
for i = 1:numel(patients)
    patientLabels{i} = cleanLabels(S.Class{patients(i)});
    patientSeconds(i) = numel(patientLabels{i});
    assert(patientSeconds(i) > 0,"A selected patient has no annotation seconds.");
end
totalSeconds = sum(patientSeconds);
XECG = zeros(contextSeconds*ecgRate,1,totalSeconds,"single");
XSpO2 = zeros(contextSeconds,1,totalSeconds,"single");
labelText = strings(totalSeconds,1);
patientID = zeros(totalSeconds,1);
patientWindowIndices = cell(numel(patients),1);

nextWindow = 1;
for i = 1:numel(patients)
    p = patients(i);
    labels = patientLabels{i};
    nSeconds = numel(labels);
    ecg = prepareECG(S.ECG{p},scalarRate(S.SR_ECG,p),ecgRate,nSeconds);
    spo2 = prepareSpO2(S.SpO2{p},scalarRate(S.SR_SpO2,p),nSeconds);

    % Pad signals, never labels. ECG padding repeats the complete endpoint
    % second; SpO2 padding repeats the endpoint value.
    ecgBySecond = reshape(ecg,ecgRate,nSeconds);
    paddedECG = [repmat(ecgBySecond(:,1),1,halfContext), ecgBySecond, ...
        repmat(ecgBySecond(:,end),1,halfContext)];
    paddedECG = paddedECG(:);
    paddedSpO2 = [repmat(spo2(1),halfContext,1); spo2; ...
        repmat(spo2(end),halfContext,1)];

    indices = nextWindow:(nextWindow+nSeconds-1);
    patientWindowIndices{i} = indices(:);
    for t = 1:nSeconds
        destination = indices(t);
        firstSample = (t-1)*ecgRate + 1;
        XECG(:,1,destination) = paddedECG(firstSample:firstSample + contextSeconds*ecgRate - 1);
        XSpO2(:,1,destination) = paddedSpO2(t:t + contextSeconds - 1);
    end
    labelText(indices) = string(labels);
    patientID(indices) = p;
    assert(numel(indices) == numel(S.Class{p}), ...
        "Windows for patient %d do not equal annotation seconds.",p);
    nextWindow = nextWindow + nSeconds;
end

Y = categorical(labelText,{"N","A"});
assert(~any(isundefined(Y)),"Class contains a label other than N or A.");
assert(isequal(string(categories(Y)),["N";"A"]), ...
    "Class order must be N followed by A.");
assert(nextWindow-1 == totalSeconds && size(XECG,3) == totalSeconds && ...
    size(XSpO2,3) == totalSeconds,"Every annotation second must have one window.");

fprintf("Total labelled seconds: %d\n",totalSeconds);
fprintf("Total windows: %d\n",size(XECG,3));
fprintf("A seconds: %d\n",sum(Y == "A"));
fprintf("N seconds: %d\n\n",sum(Y == "N"));
windowBytes = 4 * (numel(XECG) + numel(XSpO2));
fprintf("Window-array RAM (XECG + XSpO2): %.2f GiB\n\n",windowBytes/2^30);

% Instantiate and exercise both CT inputs before cross-validation starts.
sanityNet = train_cnn_windowed();
sanityScores = predict(sanityNet,dlarray(zeros(1,550,"single"),"CT"), ...
    dlarray(zeros(1,11,"single"),"CT"));
assert(numel(sanityScores) == 2 && size(sanityScores,1) == 2, ...
    "One observation must produce exactly two scores [N; A].");
clear sanityNet sanityScores

% Partition the 20 patient positions--never their seconds or windows.
rng(1);
cv = cvpartition(numel(patients),"KFold",5);
metrics = zeros(5,4);
for fold = 1:5
    trainPatients = patients(training(cv,fold));
    valPatients = patients(test(cv,fold));
    assert(isempty(intersect(trainPatients,valPatients)),"Patient leakage detected.");
    trainWindows = ismember(patientID,trainPatients);

    % All overlapping windows belonging to a patient follow that patient.
    net = train_cnn_windowed(XECG(:,:,trainWindows),XSpO2(:,:,trainWindows),Y(trainWindows));

    truth = categorical(strings(0,1),{"N","A"});
    predictedA = false(0,1);
    for i = find(ismember(patients,valPatients)).'
        idx = patientWindowIndices{i};
        probabilityA = predictProbabilityA(net,XECG(:,:,idx),XSpO2(:,:,idx),512);
        assert(numel(probabilityA) == numel(patientLabels{i}), ...
            "Validation predictions do not equal annotation seconds.");
        predictedA = [predictedA; probabilityA >= threshold]; %#ok<AGROW>
        truth = [truth; patientLabels{i}]; %#ok<AGROW>
    end

    actualA = truth == "A";
    TP = sum(predictedA & actualA); FP = sum(predictedA & ~actualA);
    FN = sum(~predictedA & actualA); TN = sum(~predictedA & ~actualA);
    sensitivity = safeDivide(TP,TP+FN);
    ppv = safeDivide(TP,TP+FP);
    f1 = safeDivide(2*TP,2*TP+FP+FN);
    accuracy = safeDivide(TP+TN,TP+FP+FN+TN);
    metrics(fold,:) = [sensitivity ppv f1 accuracy];
    fprintf("Fold %d: Sens = %.3f, PPV = %.3f, F1 = %.3f, Accuracy = %.3f\n", ...
        fold,sensitivity,ppv,f1,accuracy);
end

fprintf("\nMean Sensitivity: %.3f\n",mean(metrics(:,1)));
fprintf("Mean PPV: %.3f\n",mean(metrics(:,2)));
fprintf("Mean F1: %.3f\n",mean(metrics(:,3)));
fprintf("Mean Accuracy: %.3f\n",mean(metrics(:,4)));
fprintf("Std F1: %.3f\n",std(metrics(:,3)));

function labels = cleanLabels(raw)
labels = categorical(upper(strtrim(string(raw(:)))),{"N","A"});
assert(~any(isundefined(labels)),"Class contains a label other than N or A.");
end

function fs = scalarRate(rates,p)
if iscell(rates), fs = rates{p}; elseif isscalar(rates), fs = rates; else, fs = rates(p); end
fs = double(fs(1));
assert(isfinite(fs) && fs > 0,"Invalid sampling rate.");
end

function ecg = prepareECG(raw,fs,targetFs,nSeconds)
ecg = double(raw(:));
ecg(~isfinite(ecg)) = NaN;
ecg = fillmissing(ecg,"linear",EndValues="nearest");
assert(~isempty(ecg) && ~any(isnan(ecg)),"ECG contains no usable samples.");
[p,q] = rat(targetFs/fs,1e-12);
ecg = resample(ecg,p,q); % Polyphase resampling includes anti-alias filtering.
needed = nSeconds * targetFs;
if numel(ecg) < needed, ecg(end+1:needed) = ecg(end); end
ecg = ecg(1:needed);
centre = median(ecg);
scale = 1.4826 * median(abs(ecg-centre));
if ~isfinite(scale) || scale < eps, scale = std(ecg); end
if ~isfinite(scale) || scale < eps, scale = 1; end
ecg = single(max(-10,min(10,(ecg-centre)/scale)));
end

function spo2 = prepareSpO2(raw,fs,nSeconds)
spo2 = double(raw(:));
spo2(spo2 == 0 | ~isfinite(spo2)) = NaN;
spo2 = fillmissing(spo2,"linear",EndValues="nearest");
assert(~isempty(spo2) && ~any(isnan(spo2)),"SpO2 contains no usable samples.");
if abs(fs-1) > 1e-6
    sourceTime = (0:numel(spo2)-1)'/fs;
    spo2 = interp1(sourceTime,spo2,(0:nSeconds-1)',"linear","extrap");
end
if numel(spo2) < nSeconds, spo2(end+1:nSeconds) = spo2(end); end
spo2 = spo2(1:nSeconds);
spo2 = single(max(-2,min(2,(spo2-90)/10)));
end

function probabilityA = predictProbabilityA(net,ecg,spo2,batchSize)
n = size(ecg,3);
probabilityA = zeros(n,1,"single");
for first = 1:batchSize:n
    idx = first:min(first+batchSize-1,n);
    % Stored T-by-C-by-B arrays are permuted explicitly to C-by-T-by-B.
    ecgBatch = dlarray(permute(ecg(:,:,idx),[2 1 3]),"CTB");
    spo2Batch = dlarray(permute(spo2(:,:,idx),[2 1 3]),"CTB");
    scores = gather(extractdata(predict(net,ecgBatch,spo2Batch)));
    scores = reshape(scores,2,[]);
    assert(size(scores,2) == numel(idx),"Expected one score pair per window.");
    probabilityA(idx) = scores(2,:).';
end
end

function value = safeDivide(numerator,denominator)
if denominator == 0, value = 0; else, value = numerator/denominator; end
end
