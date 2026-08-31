function model = train_baseline_classifier(X,Y,config)
%TRAIN_BASELINE_CLASSIFIER Train-only robust scaling prevents CV leakage.
Y=categorical(Y,{'N','A'}); X=double(X);
model.imputeMedian=median(X,1,'omitnan'); model.imputeMedian(~isfinite(model.imputeMedian))=0;
X=fill(X,model.imputeMedian);
if config.classifier.normalise
    model.centre=median(X,1); model.scale=iqr(X,1); model.scale(~isfinite(model.scale)|model.scale==0)=1;
else, model.centre=zeros(1,size(X,2)); model.scale=ones(1,size(X,2)); end
X=(X-model.centre)./model.scale;
tree=templateTree('MaxNumSplits',config.classifier.maxNumSplits);
model.classifier=fitcensemble(X,Y,'Method',config.classifier.method,'Learners',tree, ...
    'NumLearningCycles',config.classifier.numLearningCycles,'LearnRate',config.classifier.learnRate,'ClassNames',categorical({'N','A'}));
model.classNames=string(model.classifier.ClassNames);
end
function X=fill(X,v), for j=1:size(X,2), bad=~isfinite(X(:,j)); X(bad,j)=v(j); end, end
