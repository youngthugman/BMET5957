function write_test_annotations(templatePath,predictionsByPatient,outputPath)
%WRITE_TEST_ANNOTATIONS Preserve cell shape and each entry's orientation; save Class only.
s=load(templatePath,'Class'); Class=s.Class; originalSize=size(Class);
if numel(Class)~=numel(predictionsByPatient), error('BMET:PatientCount','Prediction and template cell counts differ.'); end
for i=1:numel(Class)
    p=char(predictionsByPatient{i}); original=Class{i};
    if numel(p)~=numel(original), error('BMET:PredictionLength','Patient %d prediction length differs from template.',i); end
    if any(~ismember(p,['N','A'])), error('BMET:PredictionValue','Patient %d has a prediction other than N/A.',i); end
    if isrow(original), Class{i}=reshape(p,1,[]); else, Class{i}=reshape(p,[],1); end
end
assert(isequal(size(Class),originalSize)&&~any(cellfun(@(x)any(x=='?'),Class)),'Submission shape or replacement validation failed.');
save(outputPath,'Class'); vars=who('-file',outputPath); assert(isequal(vars,{'Class'}'),'Submission MAT contains extra variables.');
end
