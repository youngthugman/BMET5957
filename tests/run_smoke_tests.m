function run_smoke_tests
%RUN_SMOKE_TESTS Dataset-free checks for alignment, future context, metrics and MAT output.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(genpath(root));
config=default_config(tempname); config.spo2.preprocessing="none";
x=96*ones(41,1); x(22:25)=[95 93 92 94];
[f,n]=extract_spo2_features(x,41,config);
assert(isequal(size(f),[41 36])&&numel(n)==36&&all(isfinite(f),'all'));
rangeColumn=find(endsWith(n,"_range")); assert(f(21,rangeColumn)==4,'SpO2 range feature changed.');
xWithNaN=x; xWithNaN(1)=NaN; fWithNaN=extract_spo2_features(xWithNaN,41,config);
assert(fWithNaN(21,rangeColumn)==4,'SpO2 range did not omit NaN values.');
futureColumn=find(endsWith(n,"delta_future_5s")); assert(f(17,futureColumn)<0,'Future SpO2 context was not represented.');
[combined,names]=combine_features(ones(41,2),["e1","e2"],f,n,41); assert(isequal(size(combined),[41 38])&&numel(names)==38);
m=evaluate_predictions('AANN','ANAN'); assert(m.TP==1&&m.TN==1&&m.FP==1&&m.FN==1&&m.F1==0.5);
template=tempname+".mat"; output=tempname+".mat"; Class={repmat('?',1,3);repmat('?',2,1)}; save(template,'Class'); %#ok<NASGU>
write_test_annotations(template,{'ANA','NA'},output); z=load(output); assert(isequal(size(z.Class),[2 1])&&isrow(z.Class{1})&&iscolumn(z.Class{2}));
delete(template); delete(output); fprintf('All BMET pipeline smoke tests passed.\n');
end
