function record = load_patient_record(path, patientIndex, includeClass)
%LOAD_PATIENT_RECORD Load one patient's data from a MAT file
% without loading the whole dataset into memory.

m = matfile(path);

record.ECG  = getCell(m, 'ECG',  patientIndex);
record.SpO2 = getCell(m, 'SpO2', patientIndex);

% QRS is optional because our ECG extractor may detect peaks itself.
try
    record.QRS = getCell(m, 'QRS', patientIndex);
catch
    record.QRS = [];
end

if includeClass
    record.Class = getCell(m, 'Class', patientIndex);
else
    record.Class = [];
end

end


function value = getCell(m, name, index)
%GETCELL Retrieve one entry from a 1xN or Nx1 cell array stored in matfile.
%
% matfile does NOT allow linear indexing such as:
%     m.ECG(index)
%
% We must explicitly specify both dimensions.

sz = size(m, name);

if sz(1) == 1
    % Stored as 1 x N
    c = m.(name)(1, index);

elseif sz(2) == 1
    % Stored as N x 1
    c = m.(name)(index, 1);

else
    error( ...
        "Variable '%s' has size %s. Expected a 1xN or Nx1 cell array.", ...
        name, mat2str(sz));
end

value = c{1};

end