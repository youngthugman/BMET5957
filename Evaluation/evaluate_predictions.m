function metrics = evaluate_predictions(truth,prediction)
%EVALUATE_PREDICTIONS A is explicitly the positive class.
truth=char(truth(:)); prediction=char(prediction(:));
metrics.TP=sum(truth=='A'&prediction=='A'); metrics.TN=sum(truth=='N'&prediction=='N');
metrics.FP=sum(truth=='N'&prediction=='A'); metrics.FN=sum(truth=='A'&prediction=='N');
metrics.Sensitivity=ratio(metrics.TP,metrics.TP+metrics.FN); metrics.PPV=ratio(metrics.TP,metrics.TP+metrics.FP);
metrics.F1=ratio(2*metrics.PPV*metrics.Sensitivity,metrics.PPV+metrics.Sensitivity);
metrics.Specificity=ratio(metrics.TN,metrics.TN+metrics.FP); metrics.Accuracy=ratio(metrics.TP+metrics.TN,numel(truth));
end
function x=ratio(a,b), if b==0, x=NaN; else, x=a/b; end, end
