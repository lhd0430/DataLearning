%% Create CSV to upload to channel
% Train and Test CSV files are required.
% Test CSV contains only predictor vars.
% ThingSpeak CSV format requirements:
% - Datetime: 1/2/2017
% - field 1- 7: store predictors and response. Test data are below Train data.
% - field 8: num of train observations, num of test observations, num of var names, and var names

%% Create CSV to upload to channel
% Train and Test CSV files are required
% Response name is needed
% Test CSV contains only predictor vars
%% Combine train and test data
% Read data
train = readtable('train.csv');
test = readtable('test.csv');
% Get train and test observation num
[trainObs,~] = size(train);
[testObs,~] = size(test);
% Get test var names
testVar = test.Properties.VariableNames;
% Append test to train for each var
train(end+1:end+testObs,testVar) = test;

%% Create desiered CSV
% Create a csv file
var = {'datetime','field1','field2','field3','field4','field5','field6',...
    'field7','field8','latitude','longitude','elevation','status'};
t = table;
% Determine which methods to use
[m,n] = size(train);
if(n<=7)
    t.datetime = repmat({'Jan 2, 2017'},m,1);
    t(:,var(2:n+1)) = train;
    t(:,var(n+2:13))= cell(m,12-n);
    t{1,'field8'} = num2cell(trainObs);
    t{2,'field8'} = num2cell(testObs);
    t{3,'field8'} = num2cell(n);
    t{4:n+3,'field8'} = (train.Properties.VariableNames)';
    writetable(t,'data.csv');
else
    disp('Use MATLAB online!')
end


