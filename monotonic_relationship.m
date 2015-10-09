function[y_pclass,x_age,y_age,x_sibsp,y_sibsp,x_parch,y_parch,x_fare,y_fare]=monotonic_relationship(train_data,train_label)
%Bar plot for each numeric independent variable
%x-axis is the
%bin number, and the y-axis is the probability of survival
[no_rows,no_cols]=size(train_data);
%1.pclass
%variable already has fewer distinct values than bins, 
%Therefore we consider it already discretized.
pclass=train_data(:,1);
pclass=cell2mat(pclass);
len_pclass1=length(find(pclass==1));
len_pclass2=length(find(pclass==2));
len_pclass3=length(find(pclass==3));
pclass_1=0;
pclass_2=0;
pclass_3=0;
for row=1:no_rows
    if(isequal(train_label{row},1))
         if(isequal(train_data{row,1},1))
              pclass_1=pclass_1+1;
             elseif(isequal(train_data{row,1},2))
                  pclass_2=pclass_2+1;
                  elseif(isequal(train_data{row,1},3))
                      pclass_3=pclass_3+1;
         end
    end
end

y_pclass=[pclass_1/len_pclass1 pclass_2/len_pclass2 pclass_3/len_pclass3];
%2.age
age=cell2mat(train_data(:,4));
y=zeros(1,10);
x=zeros(1,10);
min_age=min(age);
max_age=max(age);
width=max_age-min_age;
binwidth=width/10;
start=min_age;
finish=start+binwidth;
for i=1:10
    x(1,i)=start;
    count=0;
    row_numbers=find(age>=start & age<finish);
    for j=1:length(row_numbers)
        if(isequal(train_label{row_numbers(j)},1))
           count=count+1;
        end
    end
     y(1,i)=count/length(row_numbers);
    start=finish;
    finish=start+binwidth;
end
x_age=x;
y_age=y;

%3.sibsp
sibsp=cell2mat(train_data(:,5));
values=unique(sibsp);
y=zeros(1,length(values));
x=values;
for i=1:length(values)
    count=0;
    row_numbers=find(sibsp==values(i));
    for j=1:length(row_numbers)
        if(isequal(train_label{row_numbers(j)},1))
           count=count+1;
        end
    end
     y(1,i)=count/length(row_numbers);
end
x_sibsp=x;
y_sibsp=y;
%4.parch
parch=cell2mat(train_data(:,6));
values=unique(parch);
y=zeros(1,length(values));
x=values;
for i=1:length(values)
    count=0;
    row_numbers=find(parch==values(i));
    for j=1:length(row_numbers)
        if(isequal(train_label{row_numbers(j)},1))
           count=count+1;
        end
    end
     y(1,i)=count/length(row_numbers);
end
x_parch=x;
y_parch=y;

%5.fare
fare=cell2mat(train_data(:,8));
y=zeros(1,10);
x=zeros(1,10);
min_fare=min(fare);
max_fare=max(fare);
width=max_fare-min_fare;
binwidth=width/10;
start=min_fare;
finish=start+binwidth;
for i=1:10
    x(1,i)=start;
    count=0;
    row_numbers=find(fare>=start & fare<finish);
    for j=1:length(row_numbers)
        if(isequal(train_label{row_numbers(j)},1))
           count=count+1;
        end
    end
     y(1,i)=count/length(row_numbers);
    start=finish;
    finish=start+binwidth;
end
x_fare=x;
y_fare=y;
