function info = validate_dataset(path, kind)
%VALIDATE_DATASET Fail explicitly when the MAT schema is not as expected.
arguments
    path (1,1) string
    kind (1,1) string {mustBeMember(kind,["train","test","template"])}
end
if ~isfile(path), error('BMET:MissingFile','MAT file does not exist: %s',path); end
available = string(who('-file',path));
required = "Class";
if kind ~= "template", required = ["ECG","SpO2"]; end
missing = required(~ismember(required,available));
if ~isempty(missing)
    error('BMET:DatasetSchema','%s is missing [%s]. Available variables: [%s].', ...
        path,strjoin(missing,', '),strjoin(available,', '));
end
s = whos('-file',path);
info.path = path; info.variables = available;
if kind == "template" || kind == "train"
    v = s(strcmp({s.name},'Class'));
else
    v = s(strcmp({s.name},'ECG'));
end
if ~strcmp(v.class,'cell') || numel(v.size) ~= 2
    error('BMET:DatasetSchema','Expected the patient variable in %s to be a 2-D cell array.',path);
end
info.patientArraySize = v.size; info.numPatients = prod(v.size);
if ismember("SR_ECG",available), z=load(path,'SR_ECG'); info.SR_ECG=z.SR_ECG; end
if ismember("SR_SpO2",available), z=load(path,'SR_SpO2'); info.SR_SpO2=z.SR_SpO2; end
end
