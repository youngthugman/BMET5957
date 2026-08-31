function [X,Y,patientID,featureNames,lengths] = load_cached_dataset(cacheFiles,needLabels)
%LOAD_CACHED_DATASET Assemble compact feature matrices; raw signals stay unloaded.
n=numel(cacheFiles); lengths=zeros(n,1); total=0;
for i=1:n, s=load(cacheFiles(i),'annotationLength'); lengths(i)=s.annotationLength; total=total+lengths(i); end
s=load(cacheFiles(1),'combinedFeatures','featureNames'); featureNames=s.featureNames;
X=nan(total,size(s.combinedFeatures,2),'single'); Y=repmat('?',total,1); patientID=zeros(total,1,'uint16'); cursor=1;
for i=1:n
    s=load(cacheFiles(i),'combinedFeatures','featureNames','labels');
    if ~isequal(string(s.featureNames),string(featureNames)), error('BMET:FeatureSchema','Feature schema differs in cache %s.',cacheFiles(i)); end
    idx=cursor:cursor+lengths(i)-1; X(idx,:)=single(s.combinedFeatures); patientID(idx)=i;
    if needLabels, y=char(s.labels); y=y(:); if numel(y)~=lengths(i)||any(~ismember(y,['N','A'])), error('BMET:Labels','Invalid labels in cache %s.',cacheFiles(i)); end; Y(idx)=y; end
    cursor=cursor+lengths(i);
end
end
