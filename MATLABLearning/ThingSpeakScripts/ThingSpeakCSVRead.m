%% Instruction
% Workflow:
% - Raw CSV->ThingSpeak CSV->Clear channel->Upload csv->Run app
% ThingSpeak CSV format requirements:
% - Datetime: 1/2/2017
% - field 1-7: store predictors and response. Test data are below Train data.
% - field 8: number of train observations, number of var names, and var names

%% Read data
% TODO - Replace the [] with channel ID to read data from:
readChannelID = 233743;
% TODO - Enter the Read API Key between the '' below:
readAPIKey = 'HBNM4RNXC255V64U';
% Read all data into a table
data = thingSpeakRead(readChannelID, 'ReadKey', readAPIKey, 'DateRange',...
            [datetime('Jan 1, 2017'),datetime('Jan 3, 2017')],'OutputFormat','table','Timeout',20);
% Find number of train observations and var names
trainSize = str2num(cell2mat(data{1:3,end}));
% Update table var names and remove null columns
vars = data{4:3+trainSize(3),end};
data(:,trainSize(3)+2:end) = [];
data.Properties.VariableNames = ['datetime';vars];
% Split train and test data
train = data(1:trainSize(1),:);
test = data(trainSize(1)+1:trainSize(1)+trainSize(2),:);

data

whos
%% Analysis and Visulization

