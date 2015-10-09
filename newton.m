function[accuracy]=newton(train_data,train_label)


%remove missing values
[no_rows,no_cols]=size(train_data);
x_train=train_data;
for i=1:no_cols
    rows=find(~isnan(x_train(:,i)));
    x=x_train(rows(:),:);
    x_train=x;
end

[no_rows,no_cols]=size(x_train);
%need to add a column of ones
%to the beginning of the data when not using glmfit.
col1=ones(no_rows,1);
x_train=[col1 x_train];
[no_rows,no_cols]=size(x_train);



x=(x_train);
y=(cell2mat(train_label(:,1)));
theta=zeros(no_cols,1);

%repeat until converges
for i=1:4
    output=mysigmoid(x*theta);
    accuracy(i)=cal_acc(y,output);
    z = x * theta;
    h = mysigmoid(z);
    grad = x' *(h-y);
    H = x' * diag(h) * diag(1-h) * x;
    % Calculate J (for testing convergence)
    %J(i) =sum(-y.*log(h) - (1-y).*log(1-h));
    theta = theta - H\grad;
end

function[answer]=mysigmoid(a)
answer=1./(1+exp(-a));

function[acc]=cal_acc(train_label,y_hat)
correct=0;
for i=1:length(y_hat)
            class = y_hat(i) > 0.5;
            if(class==train_label(i))
                correct=correct+1;
            end
end
acc=correct/length(y_hat);
