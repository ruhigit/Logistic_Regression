function[global_name,global_val,train_data,survived_train,test_data,survived_test]=load_data()
%Load the titanic dataset into matlab
[num,txt,raw]=xlsread('titanic3.xls');
%For each column we have to report the missing values
%Iterate over each column,and for every row that has a missing value update
%the missing value count
[no_rows,no_cols]=size(raw);
global_name={};
global_val=zeros(no_cols,1);
for col=1:no_cols
    global_name{col}=raw{1,col};
    missing_value=0;
    for row=2:no_rows-1
        check=isnan(raw{row,col});
        if(check)
            missing_value=missing_value+1;
        end
    end
    global_val(col)=missing_value;
end

%Actual data is
data=raw;
data(1,:)=[]; %Removed titles which was the first row
data(no_rows-1,:)=[]; %Remove last row as all NaN
[no_rows,no_cols]=size(data);
%Randomly shuffle data
data=data(randperm(no_rows),:);
half=ceil(no_rows/2);
train_data=data(1:half,:);
test_data=data(half+1:no_rows,:);
%Separate the survived column from the rest, as it will be the dependent
%variable we are trying predict.
survived_train=train_data(:,2);
survived_test=test_data(:,2);
train_data(:,2)=[];
test_data(:,2)=[];
