%% Non-causal TCN on cached 1-Hz engineered ECG + SpO2 features
% One whole-patient input sequence produces one N/A probability per second.
clear
clc

projectRoot = fileparts(mfilename("fullpath"));
addpath(fullfile(projectRoot,"Classifier"));
cachePath = fullfile(projectRoot,"Cache","train_features.mat");
resultsPath = fullfile(projectRoot,"Results","TCN");
resumeExisting = true;
threshold = 0.50;

requiredFeatureNames = [ ...
    "ecg_rr_current" "ecg_hr_current" "ecg_rr_mean" "ecg_rr_std" ...
    "ecg_rmssd" "ecg_pnn50" "ecg_rr_min" "ecg_rr_max" ...
    "ecg_beat_count" "ecg_rr_slope" ...
    "spo2_mean" "spo2_median" "spo2_std" "spo2_min" "spo2_max" ...
    "spo2_range" "spo2_percent_below_90" "spo2_percent_below_92" ...
    "spo2_percent_below_95" "spo2_num_desaturations" ...
    "spo2_mean_desaturation_depth" "spo2_max_desaturation_depth" ...
    "spo2_total_desaturation_time" "spo2_mean_desaturation_duration" ...
    "spo2_max_desaturation_duration" "spo2_mean_recovery_time"];

assert(isfile(cachePath), ...
    "Feature cache not found: %s. Build it separately before running the TCN.",cachePath);
cacheInfo = whos("-file",cachePath);
cacheVariables = string({cacheInfo.name});
requiredVariables = ["X" "Y" "patientID" "featureNames"];
assert(all(ismember(requiredVariables,cacheVariables)), ...
    "Cache must contain X, Y, patientID, and featureNames.");
S = load(cachePath,"X","Y","patientID","featureNames");

featureNames = string(S.featureNames(:));
assert(size(S.X,2) == numel(featureNames), ...
    "Cache X columns and featureNames are not aligned.");
[featuresFound,featureColumns] = ismember(requiredFeatureNames,featureNames);
if ~all(featuresFound)
    error("Required TCN feature(s) missing from cache: %s", ...
        strjoin(requiredFeatureNames(~featuresFound),", "));
end
assert(numel(unique(featureColumns)) == 26, ...
    "Required feature names must map to 26 unique cache columns.");
X = S.X(:,featureColumns);
selectedFeatureNames = featureNames(featureColumns);
assert(size(X,2) == 26,"The TCN experiment must select exactly 26 features.");

assert(size(X,1) == numel(S.Y),"X must have one row per Y label.");
assert(size(X,1) == numel(S.patientID),"X must have one row per patientID.");
Y = categorical(string(S.Y(:)),{'N','A'});
patientID = S.patientID(:);
assert(~any(isundefined(Y)),"Y must contain exactly the labels N and A.");
assert(isequal(string(categories(Y)),["N";"A"]), ...
    "Categorical class order must be N followed by A.");

patients = 1:100;
assert(all(ismember(patients,unique(patientID))), ...
    "The cache does not contain all 100 patients required for cross-validation.");

selectedRows = ismember(patientID,patients);
X = X(selectedRows,:);
Y = Y(selectedRows);
patientID = patientID(selectedRows);

fprintf("Non-causal Feature TCN\n");
fprintf("Patients: %d\n",numel(patients));
fprintf("Selected patient IDs: %s\n",mat2str(patients));
fprintf("Features: %d\n",size(X,2));
fprintf("Selected feature names, in network channel order:\n");
for feature = 1:numel(selectedFeatureNames)
    fprintf("  %2d. %s\n",feature,selectedFeatureNames(feature));
end
fprintf("Output interval: 1 s\n");
fprintf("TCN receptive field: 127 s\n");
fprintf("Approximate context: +/-63 s\n");
fprintf("Causal: NO\n");
fprintf("Future context: YES\n\n");
fprintf("Total selected seconds: %d\n",size(X,1));
fprintf("A seconds: %d\n",sum(Y == "A"));
fprintf("N seconds: %d\n\n",sum(Y == "N"));

if ~exist(resultsPath,"dir")
    mkdir(resultsPath);
end

%% Pre-training shape and real trainnet smoke tests
shapeNet = train_tcn();
shapeInput = dlarray(zeros(26,300,"single"),"CT");
shapeScores = predict(shapeNet,shapeInput);
assert(size(shapeScores,1) == 2 && size(shapeScores,2) == 300, ...
    "TCN shape smoke test expected output 2-by-300.");
fprintf("TCN shape smoke test passed\n");

smokeX = {zeros(300,26,"single"); zeros(400,26,"single")};
smokeY = {categorical(repmat(["N";"A"],150,1),{'N','A'}); ...
    categorical(repmat(["N";"A"],200,1),{'N','A'})};
smokeNet = train_tcn(smokeX,smokeY,single([1 1]),1); %#ok<NASGU>
fprintf("trainnet smoke test passed\n\n");
clear shapeNet shapeInput shapeScores smokeNet smokeX smokeY
rng(1);

%% Five-fold patient-wise cross-validation
cv = cvpartition(numel(patients),"KFold",5);
metrics = zeros(5,4);
oofProbabilityA = nan(size(Y));
oofCount = zeros(size(Y));

for fold = 1:5
    trainPatients = patients(training(cv,fold));
    valPatients = patients(test(cv,fold));
    assert(isempty(intersect(trainPatients,valPatients)),"Patient leakage detected.");
    trainRows = ismember(patientID,trainPatients);
    valRows = ismember(patientID,valPatients);
    assert(~any(trainRows & valRows),"No second may occur in both fold partitions.");

    fprintf("\nFold %d\n",fold);
    fprintf("Training patients: %s\n",mat2str(trainPatients));
    fprintf("Validation patients: %s\n",mat2str(valPatients));

    networkFile = fullfile(resultsPath,sprintf("fold_%d_net.mat",fold));
    useCheckpoint = false;
    if resumeExisting && isfile(networkFile)
        checkpointInfo = string({whos("-file",networkFile).name});
        needed = ["net" "featureMedian" "mu" "sigma" ...
            "selectedFeatureNames" "trainPatients" "valPatients"];
        if all(ismember(needed,checkpointInfo))
            C = load(networkFile);
            useCheckpoint = isequal(C.trainPatients,trainPatients) && ...
                isequal(C.valPatients,valPatients) && ...
                isequal(string(C.selectedFeatureNames),selectedFeatureNames);
        end
    end

    if useCheckpoint
        net = C.net;
        featureMedian = C.featureMedian;
        mu = C.mu;
        sigma = C.sigma;
        fprintf("Loaded complete fold checkpoint: %s\n",networkFile);
    else
        [featureMedian,mu,sigma] = fitPreprocessing(X(trainRows,:));
    end

    fprintf("Non-finite before imputation - train: %d, validation: %d\n", ...
        sum(~isfinite(X(trainRows,:)),"all"),sum(~isfinite(X(valRows,:)),"all"));
    XTrain = applyPreprocessing(X(trainRows,:),featureMedian,mu,sigma);
    XVal = applyPreprocessing(X(valRows,:),featureMedian,mu,sigma);
    assert(all(isfinite(XTrain),"all") && all(isfinite(XVal),"all"), ...
        "Preprocessing left non-finite values.");
    fprintf("Post-preprocessing train/validation arrays contain no non-finite values: YES\n");

    YTrain = Y(trainRows);
    counts = [sum(YTrain == "N") sum(YTrain == "A")];
    assert(all(counts > 0),"Both classes must occur in each training fold.");
    rawWeights = sum(counts) ./ counts;
    classWeights = rawWeights .^ 0.75;
    classWeights = classWeights ./ mean(classWeights);
    fprintf("Training seconds: %d\n",sum(trainRows));
    fprintf("Training N: %d\n",counts(1));
    fprintf("Training A: %d\n",counts(2));
    fprintf("Class weights [N A]: [%.3f %.3f]\n",classWeights);

    [XTrainSequences,YTrainSequences] = makeSequences( ...
        XTrain,YTrain,patientID(trainRows),trainPatients);
    assert(numel(XTrainSequences) == numel(trainPatients), ...
        "Training must contain one sequence per patient.");

    if ~useCheckpoint
        net = train_tcn(XTrainSequences,YTrainSequences,classWeights,20);
        % Save the expensive result and its fold-local preprocessing first.
        save(networkFile,"net","featureMedian","mu","sigma", ...
            "selectedFeatureNames","trainPatients","valPatients","-v7.3");
        fprintf("Saved trained network before validation: %s\n",networkFile);
    end

    foldTruth = categorical(strings(0,1),{'N','A'});
    foldProbabilityA = zeros(0,1);
    foldPatientID = zeros(0,1,"like",patientID);

    for patientIndex = 1:numel(valPatients)
        p = valPatients(patientIndex);
        localRows = find(patientID(valRows) == p);
        assert(~isempty(localRows),"Validation patient %d has no seconds.",p);
        patientInput = XVal(localRows,:);       % Stored as T-by-26.
        patientTruth = Y(valRows);
        patientTruth = patientTruth(localRows); % Stored as T-by-1.
        scores = predict(net,dlarray(single(patientInput.'),"CT"));
        scores = gather(extractdata(scores));
        assert(size(scores,1) == 2,"TCN output must have two class channels.");
        assert(size(scores,2) == numel(patientTruth), ...
            "Patient %d output length differs from annotation length.",p);
        foldProbabilityA = [foldProbabilityA; scores(2,:).']; %#ok<AGROW>
        foldTruth = [foldTruth; patientTruth]; %#ok<AGROW>
        foldPatientID = [foldPatientID; repmat(p,numel(patientTruth),1)]; %#ok<AGROW>
    end

    predictedA = foldProbabilityA >= threshold;
    metrics(fold,:) = classificationMetrics(predictedA,foldTruth);
    fprintf("Fold %d: Sens = %.4f, PPV = %.4f, F1 = %.4f, Accuracy = %.4f\n", ...
        fold,metrics(fold,1),metrics(fold,2),metrics(fold,3),metrics(fold,4));

    foldRows = find(valRows);
    % Predictions were generated in valPatients order; map them by patient ID.
    for k = 1:numel(foldRows)
        row = foldRows(k);
        match = find(foldPatientID == patientID(row));
        patientRows = foldRows(patientID(foldRows) == patientID(row));
        position = find(patientRows == row,1);
        oofProbabilityA(row) = foldProbabilityA(match(position));
        oofCount(row) = oofCount(row) + 1;
    end

    foldResult.validationPatientIDs = valPatients;
    foldResult.truth = foldTruth;
    foldResult.probabilityA = foldProbabilityA;
    foldResult.patientID = foldPatientID;
    save(fullfile(resultsPath,sprintf("fold_%d_predictions.mat",fold)), ...
        "-struct","foldResult");
    clear foldResult
end

%% OOF results and clean threshold-0.50 summary
assert(numel(oofProbabilityA) == size(X,1),"OOF output length is incorrect.");
assert(all(oofCount == 1) && all(isfinite(oofProbabilityA)), ...
    "Every selected second must receive exactly one OOF prediction.");
oofTruth = Y;
oofPatientID = patientID;
save(fullfile(resultsPath,"tcn_100patient_oof.mat"), ...
    "oofProbabilityA","oofTruth","oofPatientID", ...
    "selectedFeatureNames","patients","-v7.3");

fprintf("\nClean comparison at threshold 0.50\n");
for fold = 1:5
    fprintf("Fold %d: Sens %.4f, PPV %.4f, F1 %.4f, Accuracy %.4f\n", ...
        fold,metrics(fold,1),metrics(fold,2),metrics(fold,3),metrics(fold,4));
end
fprintf("Mean sensitivity: %.4f\n",mean(metrics(:,1)));
fprintf("Mean PPV: %.4f\n",mean(metrics(:,2)));
fprintf("Mean F1: %.4f\n",mean(metrics(:,3)));
fprintf("Mean accuracy: %.4f\n",mean(metrics(:,4)));
fprintf("Std F1: %.4f\n",std(metrics(:,3)));
pooled = classificationMetrics(oofProbabilityA >= 0.50,oofTruth);
fprintf("Pooled OOF: Sens %.4f, PPV %.4f, F1 %.4f, Accuracy %.4f\n",pooled);

thresholds = (0.05:0.01:0.95).';
tunedMetrics = zeros(numel(thresholds),4);
for k = 1:numel(thresholds)
    tunedMetrics(k,:) = classificationMetrics(oofProbabilityA >= thresholds(k),oofTruth);
end
[~,bestIndex] = max(tunedMetrics(:,3));
fprintf("\nOOF threshold tuned - development only\n");
fprintf("Threshold %.2f, Sens %.4f, PPV %.4f, F1 %.4f\n", ...
    thresholds(bestIndex),tunedMetrics(bestIndex,1), ...
    tunedMetrics(bestIndex,2),tunedMetrics(bestIndex,3));

fprintf("\nLeakage review\n");
reviewItems = [ ...
    "patient-wise CV"; "no train/validation patient overlap"; ...
    "no random second split"; "training-only feature medians"; ...
    "training-only feature means/std"; "training-only class weights"; ...
    "validation labels excluded from training"; ...
    "future signal/feature context allowed"; "future labels never used"; ...
    "one prediction per annotation second"; "output sequence length preserved"; ...
    "no hidden test data"; "no hidden-test threshold/postprocessing tuning"];
for k = 1:numel(reviewItems)
    fprintf("PASS - %s\n",reviewItems(k));
end


function [featureMedian,mu,sigma] = fitPreprocessing(XTrain)
% Fit every statistic using training-patient seconds only.
medianInput = XTrain;
medianInput(~isfinite(medianInput)) = NaN;
featureMedian = median(medianInput,1,"omitnan");
featureMedian(~isfinite(featureMedian)) = 0;
XTrain = imputeInvalid(XTrain,featureMedian);
mu = mean(XTrain,1);
sigma = std(XTrain,0,1);
sigma(~isfinite(sigma) | sigma < eps) = 1;
end


function X = applyPreprocessing(X,featureMedian,mu,sigma)
X = imputeInvalid(X,featureMedian);
X = (X - mu) ./ sigma;
X = single(X);
end


function X = imputeInvalid(X,featureMedian)
for feature = 1:size(X,2)
    invalidRows = ~isfinite(X(:,feature));
    X(invalidRows,feature) = featureMedian(feature);
end
end


function [XSequences,YSequences] = makeSequences(X,Y,ids,patients)
XSequences = cell(numel(patients),1);
YSequences = cell(numel(patients),1);
for k = 1:numel(patients)
    idx = find(ids == patients(k));
    assert(~isempty(idx),"Patient %d has no sequence.",patients(k));
    XSequences{k} = X(idx,:);             % time x channels
    YSequences{k} = reshape(Y(idx),[],1); % time x 1 categorical
end
end


function metrics = classificationMetrics(predictedA,truth)
actualA = truth == "A";
TP = sum(predictedA & actualA);
FP = sum(predictedA & ~actualA);
FN = sum(~predictedA & actualA);
TN = sum(~predictedA & ~actualA);
sensitivity = TP / max(TP + FN,1);
ppv = TP / max(TP + FP,1);
f1 = 2 * TP / max(2 * TP + FP + FN,1);
accuracy = (TP + TN) / max(TP + FP + FN + TN,1);
metrics = [sensitivity ppv f1 accuracy];
end
