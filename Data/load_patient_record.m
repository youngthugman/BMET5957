function record = load_patient_record(path, patientIndex, includeClass)
%LOAD_PATIENT_RECORD Load only one patient's cell contents where v7.3 permits.
m = matfile(path);
record.ECG = getCell(m,'ECG',patientIndex);
record.SpO2 = getCell(m,'SpO2',patientIndex);
try, record.QRS = getCell(m,'QRS',patientIndex); catch, record.QRS = []; end
if includeClass, record.Class = getCell(m,'Class',patientIndex); else, record.Class = []; end
end

function value = getCell(m,name,index)
c = m.(name)(index); value = c{1};
end
