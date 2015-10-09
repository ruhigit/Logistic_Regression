%Mutual information is commonly used for
%feature selection, where only the most important features are included in a model.
function[global_names,global_mi]=mutual_information(train_data,train_label)
global_mi=zeros(1,13);
global_names={};
%%%%%%%%%%%%%%%%%% pclass %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pclass=cell2mat(train_data(:,1));
labels=cell2mat(train_label(:,1));

%H(Y)
survived=length(find(labels(:,1)==1));
not_survived=length(find(labels(:,1)==0));
p_survived=survived/length(labels);
p_notsurvived=not_survived/length(labels);
H_Y= -p_survived*log2(p_survived)-p_notsurvived*log2(p_notsurvived);

%H(Y|X)
%For each 3 values
entropy=0;
for i=1:3
    rows=find(pclass(:)==i);
    p_pclass=length(rows)/length(pclass);
    p_surv_class=length(find(labels(rows(:))==1))/length(rows); 
    p_notsurv_class=length(find(labels(rows(:))==0))/length(rows); 
    entropy_pclass=-p_pclass*(p_surv_class*log2(p_surv_class)+ p_notsurv_class*log2( p_notsurv_class));
    entropy=entropy+entropy_pclass;
end
mi=H_Y-entropy;

global_names{1}={'pclass'};
global_mi(1)=mi;

%%%%%%%%%%%%%%%% NAME %%%%%%%%%%%%%%%%%%%%%%%%%%%%

names=train_data(:,2);
total=length(names);
labels=cell2mat(train_label(:,1));

%H(Y)
survived=length(find(labels(:,1)==1));
not_survived=length(find(labels(:,1)==0));
p_survived=survived/length(labels);
p_notsurvived=not_survived/length(labels);
H_Y= -p_survived*log2(p_survived)-p_notsurvived*log2(p_notsurvived);
mi=H_Y-0;

global_names{2}={'Name'};
global_mi(2)=mi;


%%%%%%%%%%%%%%%% SEX %%%%%%%%%%%%%%%%%%%%%%5

sex=train_data(:,3);
total=length(sex);
labels=cell2mat(train_label(:,1));

%H(Y)
survived=length(find(labels(:,1)==1));
not_survived=length(find(labels(:,1)==0));
p_survived=survived/length(labels);
p_notsurvived=not_survived/length(labels);
H_Y= -p_survived*log2(p_survived)-p_notsurvived*log2(p_notsurvived);
entropy=0;
unique_sex=unique(sex);
for i=1:length(unique_sex)
    rows=find(strcmp(sex(:),unique_sex(i)));
    p_sex=length(rows)/total;
    p_surv_sex= length(find(labels(rows(:))==1))/length(rows); 
    p_notsurv_sex= length(find(labels(rows(:))==0))/length(rows); 
    entropy=entropy-p_sex*(p_surv_sex*log2(p_surv_sex)+p_notsurv_sex*log2(p_notsurv_sex));
end
mi=H_Y-entropy;

global_names{3}={'Sex'};
global_mi(3)=mi;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  AGE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%For numeric variables, a function that discretizes them into 10 equal
%density bins (not equal width).
age=cell2mat(train_data(:,4));
labels=cell2mat(train_label(:,1));
rows=find(~isnan(age(:))); %ignore NaN values
new_age=age(rows(:));
new_labels=labels(rows(:));
%H(Y)
survived=length(find(new_labels(:,1)==1));
not_survived=length(find(new_labels(:,1)==0));
p_survived=survived/length(new_labels);
p_notsurvived=not_survived/length(new_labels);
H_Y= -p_survived*log2(p_survived)-p_notsurvived*log2(p_notsurvived);

%H(Y|X)
%sort age column
[sorted_age,rows]=sort(new_age);
labels=new_labels(rows(:));
total=length(sorted_age);
bin_size=ceil(total/10);
%Determine 10 cut points
start=0;
entropy=0;
for i=1:9
   p_age=bin_size/total;
   p_surv_age=(length(find(labels(start+1:bin_size*i,1)==1)))/bin_size;
   p_notsurv_age=length(find(labels(start+1:bin_size*i,1)==0))/bin_size;
   start=bin_size*i;
   entropy_bin= -p_age*( p_surv_age*log2( p_surv_age)+p_notsurv_age*log2(p_notsurv_age));
   entropy=entropy+entropy_bin;
end
remain=total-start;
p_age=remain/total;
p_surv_age=length(find(labels(bin_size*9+1:total,1)==1))/remain;
p_notsurv_age=length(find(labels(bin_size*9+1:total,1)==0))/remain;
entropy_bin= -p_age*( p_surv_age*log2( p_surv_age)+p_notsurv_age*log2(p_notsurv_age));
entropy=entropy+entropy_bin;
mi=H_Y-entropy;

global_names{4}={'Age'};
global_mi(4)=mi;


%%%%%%%%%%%%%%%%%% sibsp %%%%%%%%%%%%%%%%%%
sibsp=cell2mat(train_data(:,5));
labels=cell2mat(train_label(:,1));
total=length(labels);
%H(Y)
survived=length(find(labels(:,1)==1));
not_survived=length(find(labels(:,1)==0));
p_survived=survived/length(labels);
p_notsurvived=not_survived/length(labels);
H_Y= -p_survived*log2(p_survived)-p_notsurvived*log2(p_notsurvived);

%H(Y|X)
entropy=0;
unique_sibsp=unique(sibsp);
for i=1:length(unique_sibsp)
    rows=find(sibsp(:)==unique_sibsp(i));
    p=length(rows)/total;
    p_surv= length(find(labels(rows(:))==1))/length(rows); 
    p_notsurv= length(find(labels(rows(:))==0))/length(rows); 
    if(p_surv~=0 && p_notsurv~=0)
    entropy_sibsp=-p*(p_surv*log2(p_surv)+p_notsurv*log2(p_notsurv));
    entropy=entropy+entropy_sibsp;
    end
end
mi=H_Y-entropy;

global_names{5}={'Sibsp'};
global_mi(5)=mi;



%%%%%%%%%%%%%%%%%% parch %%%%%%%%%%%%%%%%%%
parch=cell2mat(train_data(:,6));
labels=cell2mat(train_label(:,1));
total=length(labels);
%H(Y)
survived=length(find(labels(:,1)==1));
not_survived=length(find(labels(:,1)==0));
p_survived=survived/length(labels);
p_notsurvived=not_survived/length(labels);
H_Y= -p_survived*log2(p_survived)-p_notsurvived*log2(p_notsurvived);

%H(Y|X)
entropy=0;
unique_parch=unique(parch);
for i=1:length(unique_parch)
    rows=find(parch(:)==unique_parch(i));
    p=length(rows)/total;
    p_surv= length(find(labels(rows(:))==1))/length(rows); 
    p_notsurv= length(find(labels(rows(:))==0))/length(rows); 
    if(p_surv~=0 && p_notsurv~=0)
    entropy_parch=-p*(p_surv*log2(p_surv)+p_notsurv*log2(p_notsurv));
    entropy=entropy+entropy_parch;
    end
end
mi=H_Y-entropy;

global_names{6}={'Parch'};
global_mi(6)=mi;


%%%%%%% ticket %%%%%%%%%%%

names=train_data(:,7);
labels=cell2mat(train_label(:,1));
entropy=0;
%H(Y)
survived=length(find(labels(:,1)==1));
not_survived=length(find(labels(:,1)==0));
p_survived=survived/length(labels);
p_notsurvived=not_survived/length(labels);
H_Y= -p_survived*log2(p_survived)-p_notsurvived*log2(p_notsurvived);
total=length(names);
for i=1:length(names)
    rows=find(strcmp(names(:),names(i)));
    if(~rows)
        rows(1)=i;
    end
    p=length(rows)/total;
    p_surv= length(find(labels(rows(:))==1))/length(rows);
    p_notsurv= length(find(labels(rows(:))==0))/length(rows);
    if(p_surv~=0 && p_notsurv~=0 && ~isnan(p_surv) && ~isnan(p_notsurv) )
    e=-p*(p_surv*log2(p_surv)+p_notsurv*log2(p_notsurv));
    entropy=entropy+e;
    end
end
mi=H_Y-entropy;
global_names{7}={'Ticket'};
global_mi(7)=mi;


%%%%%%%%%%%%%%% fare %%%%%%%%%%%%%%

%For numeric variables, a function that discretizes them into 10 equal
%density bins (not equal width).
fare=cell2mat(train_data(:,8));
labels=cell2mat(train_label(:,1));
rows=find(~isnan(fare(:))); %ignore NaN values
new_fare=fare(rows(:));
new_labels=labels(rows(:));
%H(Y)
survived=length(find(new_labels(:,1)==1));
not_survived=length(find(new_labels(:,1)==0));
p_survived=survived/length(new_labels);
p_notsurvived=not_survived/length(new_labels);
H_Y= -p_survived*log2(p_survived)-p_notsurvived*log2(p_notsurvived);

%H(Y|X)
%sort fare column
[sorted_fare,rows]=sort(new_fare);
labels=new_labels(rows(:));
total=length(sorted_fare);
bin_size=floor(total/10);
%Determine 10 cut points
start=0;
entropy=0;
for i=1:9
   p_age=bin_size/total;
   p_surv_age=(length(find(labels(start+1:bin_size*i,1)==1)))/bin_size;
   p_notsurv_age=length(find(labels(start+1:bin_size*i,1)==0))/bin_size;
   start=bin_size*i;
   entropy_bin= -p_age*( p_surv_age*log2( p_surv_age)+p_notsurv_age*log2(p_notsurv_age));
   entropy=entropy+entropy_bin;
end
remain=total-(start+1);
p_age=remain/total;
p_surv_age=length(find(labels(start+1:total,1)==1))/remain;
p_notsurv_age=length(find(labels(bin_size*9+1:total,1)==0))/remain;
entropy_bin= -p_age*( p_surv_age*log2( p_surv_age)+p_notsurv_age*log2(p_notsurv_age));
entropy=entropy+entropy_bin;
mi=H_Y-entropy;

global_names{8}={'Fare'};
global_mi(8)=mi;

%%%%% cabin %%%%%%%%%%
cabin=train_data(:,9);
labels=cell2mat(train_label(:,1));
j=1;
for i=1:length(cabin)
    check=isnan(cabin{i});
    if(check)
           continue;
    end
    rows(j)=i;
    j=j+1;
end
cabin=cabin(rows(:));
labels=labels(rows(:));
%H(Y)
survived=length(find(labels(:,1)==1));
not_survived=length(find(labels(:,1)==0));
p_survived=survived/length(labels);
p_notsurvived=not_survived/length(labels);
H_Y= -p_survived*log2(p_survived)-p_notsurvived*log2(p_notsurvived);
entropy=0;
total=length(cabin);
for i=1:total
    rows=find(strcmp(cabin(:),cabin(i)));
    p=length(rows)/total;
    p_surv= length(find(labels(rows(:))==1))/length(rows);
    p_notsurv= length(find(labels(rows(:))==0))/length(rows);
    if(p_surv~=0 && p_notsurv~=0 && ~isnan(p_surv) && ~isnan(p_notsurv) )
    entropy=entropy-p*(p_surv*log2(p_surv)+p_notsurv*log2(p_notsurv));
    end
end
mi=H_Y-entropy;

global_names{9}={'Cabin'};
global_mi(9)=mi;

%%%%%%%%%%%%%% embarked %%%%%%%%%%%%

sex=train_data(:,10);
total=length(sex);
labels=cell2mat(train_label(:,1));
rows=find(strcmp(sex(:),'S') | strcmp(sex(:),'C')| strcmp(sex(:),'Q')); %ignore NaN values
sex=sex(rows(:));
labels=labels(rows(:));
%H(Y)
survived=length(find(labels(:,1)==1));
not_survived=length(find(labels(:,1)==0));
p_survived=survived/length(labels);
p_notsurvived=not_survived/length(labels);
H_Y= -p_survived*log2(p_survived)-p_notsurvived*log2(p_notsurvived);
entropy=0;
unique_sex=unique(sex);
for i=1:length(unique_sex)
    rows=find(strcmp(sex(:),unique_sex(i)));
    p_sex=length(rows)/total;
    p_surv_sex= length(find(labels(rows(:))==1))/length(rows); 
    p_notsurv_sex= length(find(labels(rows(:))==0))/length(rows); 
    entropy=entropy-p_sex*(p_surv_sex*log2(p_surv_sex)+p_notsurv_sex*log2(p_notsurv_sex));
end
mi=H_Y-entropy;

global_names{10}={'Embarked'};
global_mi(10)=mi;
%%%%% boat %%%%%%%%%%
data=train_data(:,11);
labels=cell2mat(train_label(:,1));
j=1;
for i=1:length(data)
    check=isnan(data{i});
    if(check)
           continue;
    end
    rows(j)=i;
    j=j+1;
end
data=data(rows(:));
labels=labels(rows(:));
%H(Y)
survived=length(find(labels(:,1)==1));
not_survived=length(find(labels(:,1)==0));
p_survived=survived/length(labels);
p_notsurvived=not_survived/length(labels);
H_Y= -p_survived*log2(p_survived)-p_notsurvived*log2(p_notsurvived);
entropy=0;
total=length(data);
for i=1:total
    rows=find(strcmp(data(:),data(i)));
    if(~rows)
        rows(1)=i;
    end
    p=length(rows)/total;
    p_surv= length(find(labels(rows(:))==1))/length(rows);
    p_notsurv= length(find(labels(rows(:))==0))/length(rows);
    if(p_surv~=0 && p_notsurv~=0 && ~isnan(p_surv) && ~isnan(p_notsurv) )
    entropy=entropy-p*(p_surv*log2(p_surv)+p_notsurv*log2(p_notsurv));
    end
end
mi=H_Y-entropy;
global_names{11}={'Boat'};
global_mi(11)=mi;

%%%%% body %%%%%%%%%%
data=train_data(:,12);
labels=cell2mat(train_label(:,1));
j=1;
for i=1:length(data)
    check=isnan(data{i});
    if(check)
           continue;
    end
    rows(j)=i;
    j=j+1;
end
data=data(rows(:));
labels=labels(rows(:));
%H(Y)
H_Y=0;
survived=length(find(labels(:,1)==1));
not_survived=length(find(labels(:,1)==0));
p_survived=survived/length(labels);
p_notsurvived=not_survived/length(labels);
if(p_survived~=0 && p_notsurvived~=0 && ~isnan(p_survived) && ~isnan(p_notsurvived) )
H_Y= -p_survived*log2(p_survived)-p_notsurvived*log2(p_notsurvived);
end
entropy=0;
total=length(data);
for i=1:total
    rows=i;
    p=length(rows)/total;
    p_surv= length(find(labels(rows(:))==1))/length(rows);
    p_notsurv= length(find(labels(rows(:))==0))/length(rows);
    if(p_surv~=0 && p_notsurv~=0 && ~isnan(p_surv) && ~isnan(p_notsurv) )
    entropy=entropy-p*(p_surv*log2(p_surv)+p_notsurv*log2(p_notsurv));
    end
end
mi=H_Y-entropy;

global_names{12}={'body'};
global_mi(12)=mi;

%%%%%%%%%55 home dest %%%%%%55
data=train_data(:,13);
labels=cell2mat(train_label(:,1));
j=1;
for i=1:length(data)
    check=isnan(data{i});
    if(check)
           continue;
    end
    rows(j)=i;
    j=j+1;
end
data=data(rows(:));
labels=labels(rows(:));
%H(Y)
H_Y=0;
survived=length(find(labels(:,1)==1));
not_survived=length(find(labels(:,1)==0));
p_survived=survived/length(labels);
p_notsurvived=not_survived/length(labels);
 if(p_survived~=0 && p_notsurvived~=0 && ~isnan(p_survived) && ~isnan(p_notsurvived) )
H_Y= -p_survived*log2(p_survived)-p_notsurvived*log2(p_notsurvived);
 end
entropy=0;
uniq=unique(data);
total=length(data);
for i=1:length(uniq)
    rows=find(strcmp(data(:),uniq(i)));
    p=length(rows)/total;
    p_surv= length(find(labels(rows(:))==1))/length(rows);
    p_notsurv= length(find(labels(rows(:))==0))/length(rows);
    if(p_surv~=0 && p_notsurv~=0 && ~isnan(p_surv) && ~isnan(p_notsurv) )
    e=-p*(p_surv*log2(p_surv)+p_notsurv*log2(p_notsurv));
    entropy=entropy+e;
    end
end
mi=H_Y-entropy;

global_names{13}={'Home Dest'};
global_mi(13)=mi;


