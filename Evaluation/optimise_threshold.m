function [bestThreshold,tableOut] = optimise_threshold(truth,scores,grid)
%OPTIMISE_THRESHOLD Maximise held-out F1; deterministic tie-break picks lowest threshold.
f1=nan(numel(grid),1);
for i=1:numel(grid), p=repmat('N',numel(truth),1); p(scores>=grid(i))='A'; m=evaluate_predictions(truth,p); f1(i)=m.F1; end
[~,i]=max(f1); bestThreshold=grid(i); tableOut=table(grid(:),f1,'VariableNames',{'Threshold','F1'});
end
