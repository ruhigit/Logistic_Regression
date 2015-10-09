function[train_accu,test_accu,accuracy_t,accuracy,x_training,x_testing]=missing_values(train_data,train_label,test_data,test_label)
%Create separate X matrices for both the training and testing
%data out of the columns 



%{sex, pclass, fare, embarked_1, embarked_2, parch, sibsp, age}
%   1      2       3    4           5           6     7     8 


x_training=matrix(train_data);
x_testing=matrix(test_data);

%normalize x_train and x_test
for i=1:7
    avg(i)=sum(x_training(:,i))/length(x_training);
    std_dev(i)= (sum((x_training(:,i)-avg(i)).^2))/(length(x_training)-1);
    x_train(:,i)=(x_training(:,i)-avg(i))/sqrt(std_dev(i));
    x_test(:,i)=(x_testing(:,i)-avg(i))/sqrt(std_dev(i));
end
%for age
rows=find(~isnan(x_training(:,8)));
non_nan_age=x_training(rows(:),8);  %remove NaN entries for avg and std_dev calculation
avg(8)=sum(non_nan_age(:,1))/length(x_training);
std_dev(8)= (sum((non_nan_age(:,1)-avg(8)).^2))/(length(x_training)-1);
x_train(:,8)=(x_training(:,8)-avg(8))/sqrt(std_dev(8));
x_test(:,8)=(x_testing(:,8)-avg(8))/sqrt(std_dev(8));

train_label=cell2mat(train_label(:,1));
test_label=cell2mat(test_label(:,1));
%%%%%%%%%%%%%   Multiple Models:   %%%%%%%%%%%%
disp('%%% Multiple Models %%%');
%1.Model using the column age
model1 = glmfit(x_train, train_label, 'binomial');
%2. Model leaving out the age column
x_train_1=x_train(:,1:7);
model2 = glmfit(x_train_1, train_label, 'binomial');

%%%% training accuracy %%%%%
train_accu=cal_accu(x_train,train_label,model1,model2);
disp('Training accuracy');
disp(train_accu);
%%%% test accuracy %%%%%
test_accu=cal_accu(x_test,test_label,model1,model2);
disp('Testing accuracy');
disp(test_accu);

%%%%%%% substituting values %%%%%%%%
%Training data
%Replace NaN with avg age values
size_train=size(x_train);
x_train_avg=zeros(size_train);
for i=1:length(x_train)
    x_train_avg(i,1:7)=x_train(i,1:7);
    if(isnan(x_train(i,8)))
       x_train_avg(i,8)=avg(8);
    else
        x_train_avg(i,8)=x_train(i,8);
    end
end
%Test data
%Replace NaN with avg age values
size_test=size(x_test);
x_test_avg=zeros(size_test);
for i=1:length(x_test)
    x_test_avg(i,1:7)=x_test(i,1:7);
    if(isnan(x_test(i,8)))
       x_test_avg(i,8)=avg(8);
    else
        x_test_avg(i,8)=x_test(i,8);
    end
end

%normalize x_train_avg and x_test_avg
for i=1:8
    avg(i)=sum(x_train_avg(:,i))/length(x_train_avg);
    std_dev(i)= (sum((x_train_avg(:,i)-avg(i)).^2))/(length(x_train_avg)-1);
    x_train_avg(:,i)=(x_train_avg(:,i)-avg(i))/sqrt(std_dev(i));
    x_test_avg(:,i)=(x_test_avg(:,i)-avg(i))/sqrt(std_dev(i));
end
disp('Substituting values');
%1.Model using the column age
model = glmfit(x_train_avg, train_label, 'binomial');
y_hat = glmval(model, x_train_avg, 'logit');
correct=0;
%Trainig accuracy
for i=1:length(y_hat)
    class = y_hat(i) > 0.5;
    if(class==train_label(i))
        correct=correct+1;
    end
end
accuracy_t=correct/length(train_label);
disp('Training accuracy');
disp(accuracy_t);

y_hat = glmval(model, x_test_avg, 'logit');
correct=0;
%Test accuracy
for i=1:length(y_hat)
    class = y_hat(i) > 0.5;
    if(class==test_label(i))
        correct=correct+1;
    end
end
accuracy=correct/length(test_label);
disp('test accuracy');
disp(accuracy);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[accuracy]=cal_accu(x_train,train_label,model1,model2)
%%% ip= x_train%%%
%%% Divide the training set into w1 and w2. 
%%%w1 has age column, w2 age column is NaN.
rows_age=find(~isnan(x_train(:,8)));
w1=x_train(rows_age(:),:);
y1=train_label(rows_age(:));

rows_nonage=find(isnan(x_train(:,8)));
w2=x_train(rows_nonage(:),1:7);
y2=train_label(rows_nonage(:));

%%%w1 will be trained using model1 and w2 using
%%% model2
y_hat1 = glmval(model1, w1, 'logit');
y_hat2 = glmval(model2, w2, 'logit');

correct=0;
%Trainig accuracy
for i=1:length(y_hat1)
    class = y_hat1(i) > 0.5;
    if(class==y1(i))
        correct=correct+1;
    end
end
for i=1:length(y_hat2)
    class = y_hat2(i) > 0.5;
    if(class==y2(i))
        correct=correct+1;
    end
end
accuracy=correct/length(train_label);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[x]=matrix(data)
size=length(data);
x=zeros(size,8);
%Iterate through training data and fill x_train
for row=1:size
    %sex
    %male-1
    %female-0
    if(isequal(data{row,3},'male'))
        x(row,1)=1;
    elseif(isequal(data{row,3},'female'))
        x(row,1)=0;
    end
    %pclass
    x(row,2)=data{row,1};
    %fare
    if(isnan(data{row,8}))
        x(row,3)=33.2955;
    else
         x(row,3)=data{row,8};
    end
   
    %embarked
    if(isequal(data{row,10},'C'))
        x(row,4)=0;
        x(row,5)=0;
    elseif(isequal(data{row,10},'Q'))
        x(row,4)=0;
        x(row,5)=1;
    else
        x(row,4)=1;
        x(row,5)=0;
    end
    %parch
    x(row,6)=data{row,6};
    %sibsp
    x(row,7)=data{row,5};
    %age
     x(row,8)=data{row,4};
end