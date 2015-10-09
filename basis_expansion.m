function[basis_return,X_train,X_test]=basis_expansion(x_train,x_test)

%x_train and x_test are matrices that still have NaN values when
%age is missing.
%{sex, pclass, fare, embarked_1, embarked_2, parch, sibsp, age}
%   1      2       3    4           5           6     7     8 


%Append columns with the square-root of the numeric independent variables 
%(pclass, age, sibsp,parch, fare) 
%to both the training and testing X matrices.

x_train1=append_13columns(x_train);
x_test1=append_13columns(x_test);

%Discretize the 5 numeric variables from the training data 
%into at most 10 equal density bins
cutpoints=zeros(5,11);

%for pclass
values=unique(x_train(:,2)); 
cutpoints(1,1:length(values))=values;
%for age
rows=find(~isnan(x_train(:,8)));
temp_age=x_train(rows(:),8);
temp_age=sort(temp_age);
density=ceil(length(temp_age)/10);
cutpoints(2,1)=min(temp_age);
for i=1:9
    cutpoints(2,i+1)=temp_age(i*density);
end
cutpoints(2,11)=max(temp_age);
%for sibsp
values=unique(x_train(:,7)); 
cutpoints(3,1:length(values))=values;
%for parch
values=unique(x_train(:,6)); 
cutpoints(4,1:length(values))=values;
%for fare
temp_fare=x_train(rows(:),3);
temp_fare=sort(temp_fare);
density=ceil(length(temp_fare)/10);
cutpoints(5,1)=min(temp_fare);
for i=1:9
    cutpoints(5,i+1)=temp_fare(i*density);
end
cutpoints(5,11)=max(temp_fare);

%Treat the discretized numeric variables as
%categorical variables, encode them with dummy variables and append them to the X matrices.
x_train_disc=discretized_matrix(x_train,cutpoints);
x_test_disc=discretized_matrix(x_test,cutpoints);

[encoding_train,encoding_test]=encode(x_train_disc,x_test_disc);
x_train_final=[x_train1 encoding_train];
x_test_final=[x_test1 encoding_test];


%Append interaction variables for all pairwise combinations of columns currently in the X
%matrices.
interaction_matrix_train=interaction(x_train_final);
interaction_matrix_test=interaction(x_test_final);

[X_train,X_test]=final(interaction_matrix_train,interaction_matrix_test);
disp('No of columns');
no_cols=size(X_train,2);
disp(no_cols);
basis_return=no_cols;
%%% normalize %%%
for i=1:no_cols
    rows=find(~isnan(X_train(:,i)));
    x_training=X_train(rows(:),i);  %remove NaN entries for avg and std_dev calculation
    avg(i)=sum(x_training(:,1))/length(x_training);
    std_dev(i)= (sum((x_training(:,1)-avg(i)).^2))/(length(x_training)-1);
    X_train(:,i)=(X_train(:,i)-avg(i))/sqrt(std_dev(i));
    X_test(:,i)=(X_test(:,i)-avg(i))/sqrt(std_dev(i));
end


%delete any columns that
%have less than 2 distinct non-NaN values in the training data
function[X,X_1]=final(interaction_matrix,interaction_matrix1)
X=[];
X_1=[];
no_cols=(size(interaction_matrix,2));
for col=1:no_cols
    %non-nan rows
    rows=find(~isnan(interaction_matrix(:,col)));
    unique_values=unique(interaction_matrix(rows(:),col));
    if(length(unique_values)>=2)
      X=[X interaction_matrix(:,col)];
      X_1=[X_1 interaction_matrix1(:,col)];
    end
end


function[interaction_matrix]=interaction(x_train_final)
%X(:,i) .* X(:,j) for all i and j, such that j < i.
[no_rows,no_columns]=size(x_train_final);
interaction_matrix=[];
for i=no_columns:-1:2
    for j=1:i-1
        temp=x_train_final(:,i).*x_train_final(:,j);
        interaction_matrix=[interaction_matrix temp];
    end
end


function[x_train1]=append_13columns(x_train)
%training x matrix
[train_rows,train_cols]=size(x_train);
x_train1=zeros(train_rows,train_cols+5);
x_train1(:,1:train_cols)=x_train(:,1:train_cols);  % first 8 columns as it is
x_train1(:,train_cols+1)=sqrt(x_train(:,2)); % sqrt of pclass
x_train1(:,train_cols+2)=sqrt(x_train(:,8)); % sqrt of age
x_train1(:,train_cols+3)=sqrt(x_train(:,7)); % sqrt of sibsp
x_train1(:,train_cols+4)=sqrt(x_train(:,6)); % sqrt of parch
x_train1(:,train_cols+5)=sqrt(x_train(:,3)); % sqrt of fare


function[x_train2]=discretized_matrix(x_train,cutpoints)
%(pclass, age, sibsp,parch, fare)
no_rows=length(x_train);
x_train2=zeros(no_rows,5);
x_train2(:,1)=x_train(:,2); %append pclass as it is
%for age
%find all rows belonging to bin_n and replace by n
for i=1:10
    start=cutpoints(2,i);
    finish=cutpoints(2,i+1);
    rows=find(x_train(:,8)>=start & x_train(:,8)<finish);
    x_train2(rows(:),2)=i;
end
x_train2(:,3)=x_train(:,7); %append sibsp as it is
x_train2(:,4)=x_train(:,6); %append parch as it is
%for fare
%find all rows belonging to bin_n and replace by n
for i=1:10
    start=cutpoints(5,i);
    finish=cutpoints(5,i+1);
    rows=find(x_train(:,3)>=start & x_train(:,3)<finish);
    x_train2(rows(:),5)=i;
end

function[encoding_matrix_train,encoding_matrix_test]=encode(x_train_disc,x_test_disc)
%encode the 5 columns to k-1 encoding
[no_rows_tr,no_columns_tr]=size(x_train_disc);
[no_rows_te,no_columns_te]=size(x_test_disc);
%start=1;
%total_cols=0;
encoding_matrix_train=[];
encoding_matrix_test=[];
for col=1:no_columns_tr
    unique_values_train=unique(x_train_disc(:,col));
    unique_values_test=unique(x_test_disc(:,col));
    unique_values=union(unique_values_train,unique_values_test);
    encoding_length=length(unique_values)-1;
    temp_matrix_tr=zeros(no_rows_tr,encoding_length);
    temp_matrix_te=zeros(no_rows_te,encoding_length);
    %total_cols=total_cols+encoding_length;
    for i=1:length(unique_values)
        value=unique_values(i);
        rows_tr=find(x_train_disc(:,col)==value); %all rows having a particular value
        rows_te=find(x_test_disc(:,col)==value); %all rows having a particular value
        col_tobe_1=i-1;
        if(col_tobe_1~=0)
            temp_matrix_tr(rows_tr(:),col_tobe_1)=1;
            temp_matrix_te(rows_te(:),col_tobe_1)=1;
        end
    end
    encoding_matrix_train=[encoding_matrix_train temp_matrix_tr];
    encoding_matrix_test=[encoding_matrix_test temp_matrix_te];
    %start=total_cols+1;
end