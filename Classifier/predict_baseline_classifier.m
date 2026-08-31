function [labels,apnoeaScore] = predict_baseline_classifier(model,X,threshold)
%PREDICT_BASELINE_CLASSIFIER Select the A score by class name, never column position assumptions.
X=double(X); for j=1:size(X,2), bad=~isfinite(X(:,j)); X(bad,j)=model.imputeMedian(j); end
X=(X-model.centre)./model.scale; [~,scores]=predict(model.classifier,X);
aColumn=find(model.classNames=="A",1); if isempty(aColumn), error('BMET:PositiveClass','Classifier has no A score column.'); end
% Boosted-ensemble margins are unbounded. This monotonic logistic mapping
% makes threshold grids interpretable while preserving score ranking.
margin=scores(:,aColumn); apnoeaScore=1./(1+exp(-2*margin));
labels=repmat('N',size(X,1),1); labels(apnoeaScore>=threshold)='A';
end
