function result = patientwise_cross_validation(X,Y,patientID,config)
%PATIENTWISE_CROSS_VALIDATION Assign whole patients, never seconds, to folds.
patients=unique(patientID); rng(config.randomSeed); order=patients(randperm(numel(patients))); foldOf=zeros(max(patients),1);
for i=1:numel(order), foldOf(order(i))=mod(i-1,config.crossValidation.numFolds)+1; end
scores=nan(numel(Y),1); foldMetrics=table;
for f=1:config.crossValidation.numFolds
    valid=foldOf(double(patientID))==f; train=~valid;
    assert(isempty(intersect(unique(patientID(train)),unique(patientID(valid)))),'Patient leakage detected.');
    fprintf('CV fold %d/%d: %d training patients, %d validation patients\n',f,config.crossValidation.numFolds,numel(unique(patientID(train))),numel(unique(patientID(valid))));
    model=train_baseline_classifier(X(train,:),Y(train),config); [~,scores(valid)]=predict_baseline_classifier(model,X(valid,:),0.5);
end
[threshold,thresholdTable]=optimise_threshold(Y,scores,config.thresholdGrid); p=repmat('N',numel(Y),1); p(scores>=threshold)='A';
for f=1:config.crossValidation.numFolds
    valid=foldOf(double(patientID))==f; m=evaluate_predictions(Y(valid),p(valid)); row=struct2table(m); row.Fold=f; foldMetrics=[foldMetrics;row]; %#ok<AGROW>
    fprintf('  fold %d at threshold %.2f: sensitivity %.3f, PPV %.3f, F1 %.3f\n',f,threshold,m.Sensitivity,m.PPV,m.F1);
end
pooled=evaluate_predictions(Y,p);
result.scores=scores; result.predictions=p; result.foldMetrics=foldMetrics; result.pooledMetrics=pooled; result.threshold=threshold; result.thresholdTable=thresholdTable; result.patientFold=foldOf;
end
