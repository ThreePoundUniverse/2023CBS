function [List,FileC,List_name,List_all,Subj,SubjC]=get_file_I_wantv1_2(direction,suffix)
    List=dir(direction);List(1:2,:)=[];
    for i=size(List,1):-1:1 % 从list中删掉文件夹
        if List(i).isdir==1
            List(i)=[];
        end
    end
    for i=size(List,1):-1:1
        if ~strcmp(List(i).name(end-length(suffix)+1:end),suffix)
            List(i,:)=[];
        end
    end
    FileC=size(List,1);
    
    List_name=cell(FileC,1);
    for i=1:FileC
        idown=findstr(List(i).name,"_");
        Subj{i}=List(i).name(1: (idown(1)-1) );
        List_name{i}=List(i).name(1:end-5);
    end
    Subj=unique(Subj)';
    SubjC=length(Subj);
    
    % 整理Subj汇总
    List_all(:,1)=List_name;
    SI_count=zeros(SubjC,1);
    for iC=1:FileC
        idown=findstr(List(iC).name,"_");
        SjNm=List(iC).name(1: (idown(1)-1) );
        List_all{iC,2}=SjNm;
        SI=find(strcmp(Subj,SjNm));
        SI_count(SI)=SI_count(SI)+1;
        List_all{iC,3}=SI_count(SI);
    end
    
    
    
    
end