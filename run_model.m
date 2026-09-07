%% This file is where we orchestrate our model  
projectRoot = fileparts(mfilename('fullpath'));
addpath(genpath(projectRoot));

trainPath = "E:\Desktop\Downloads\ProjectTrainData.mat";

% Fixed whole-patient development subset for ECG-window comparisons.
rng(1);
patients = sort(randperm(100,20));

fprintf('ECG window experiment\n');
fprintf('ECG window: 60 s\n');
fprintf('ECG features: 125\n');
fprintf('SpO2 features: 16\n');
fprintf('Total features: 141\n');
fprintf('Patients: 20\n');
fprintf('Output interval: 1 s\n');
fprintf('Selected patient IDs: %s\n',sprintf('%d ',patients));

[X, Y, patientID, featureNames] = build_features(trainPath, patients);
fprintf('size(X): [%d %d]\n',size(X,1),size(X,2));
fprintf('size(Y): [%d %d]\n',size(Y,1),size(Y,2));
fprintf('size(patientID): [%d %d]\n',size(patientID,1),size(patientID,2));

nonFinite = ~isfinite(X);
fprintf('Rows with any non-finite feature: %d\n',sum(any(nonFinite,2)));
ecgNonFiniteCounts = sum(nonFinite(:,1:125),1);
[worstCounts,worstIndices] = sort(ecgNonFiniteCounts,'descend');
numToPrint = min(10,sum(worstCounts > 0));
fprintf('Worst affected ECG features (non-finite rows):\n');
if numToPrint == 0
    fprintf('  none\n');
else
    for k = 1:numToPrint
        fprintf('  %s: %d\n',featureNames(worstIndices(k)),worstCounts(k));
    end
end

[model, results] = train_classifier(X, Y, patientID);
