%% This file is where we orchestrate our model  

trainPath = "C:\Users\maxas\Downloads\ProjectTrainData_ML.mat";

patients = 1; % can add a specific # of patients if we want.
[X, Y, patientID, featureNames] = build_features(trainPath, patients);
disp(size(X));
disp(size(Y));
disp(size(patientID));
disp(featureNames);

