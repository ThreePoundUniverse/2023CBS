function main_pipelineFin_hh2()
% fnirs pre-processing pipeline
% by lsc 2021/12
% change yjy 2022/12
rng(1);
%% load nirs
Folder='task_new'; % nirs data
Folder_For_Save = 'data_out_yzb';
[List,FileC,List_name,List_all,Subj,SubjC]=get_file_I_wantv1_2(fullfile(pwd,Folder),'nirs');

tic;
for iSubj=1:SubjC
    SubjFile=find( strcmp(List_all(:,2),Subj(iSubj)) ==1 );
    SubjFileCount=length(SubjFile);
    dcNbf_splice=[];  % Ԥ������HbX����
    t_splice=[]; tInc0_splice=[]; tIncCh_splice=[];
    s_splice=[];
    chan_asag_rej=true(40,1);
    
    for iFile=1:SubjFileCount
        % for iFile=1:2
        %% processing pipeline
        % setting
        LoadHint=strcat('loaded...',List(SubjFile(iFile)).name);
        disp(LoadHint);
        
        % ��nirs�ļ�
        nirs_file_name = List(SubjFile(iFile)).name;
        load(fullfile(pwd,Folder, nirs_file_name),'-mat' );
        if ~isempty(t_splice)
            t=t+t_splice(end);  % ʱ��t����
        end
        SD.MeasListAct=ones(length(SD.MeasList),1);
        fs=round(1/(t(2)-t(1)));
        tInc=zeros(length(t),1);

        
        % ********** Motion Correct Test **********
%         p = 0.99;
%         turnon = 1;
%         tInc=zeros(40, length(t));
%         t_idx = (t >= 2680) & (t < 2690);
%         tInc(:, t_idx) = 1;
%          dod = hmrConc2OD(dc_raw, SD, [6 6 6] );%��任��Ѫ��Ũ�ȱ�Ϊ���ܶ�
%          dodKWavelet = hmrMotionCorrectKurtosisWavelet(dod,SD,3.3);
%          dc = hmrOD2Conc(dodKWavelet, SD, [6 6 6]);
         dc = dc_raw;
         % �˲�
         dcf=filt_nirsLB(dc,fs);
         % PCAȥ�˶�α����ʱ��PCA��
%          tIncMan=[];tMotion=2;tMask=2;std_thresh=15;amp_thresh=0.5;nSV=0.97;maxIter=5;
%          [dodN1,tInc,svs,nSV,tInc0] = hmrMotionCorrectPCArecurse(dod, fs, SD, tIncMan, tMotion, tMask, std_thresh, amp_thresh, nSV, maxIter, 1);   
         
         % PCAȥȫ���������ռ�PCA��
%          nSV1=0.9;  % ���������ɷ�����
%          [dodNf, ~, ~] = hmrPCAFilter(dod, SD, nSV1); % �ռ�PCA
%          dcf = hmrOD2Conc(dodNf,SD,[6 6 6]);
             
        % ���ܳɷַ���
        [~,dcNbf,~] = hdms(dcf(:,1:2,:));
%         dcNbf = dcf(:,1:2,:);
        
%         LoadHint=strcat(List(SubjFile(iFile)).name,'has',a,'channels passed');
%         disp(LoadHint);
        nirs_file_name=[nirs_file_name(1:end-5),'_1'];
        save_name = fullfile(pwd, Folder_For_Save, nirs_file_name);
        save(save_name, 'dcNbf','aux','s','t','tIncMan','SD','dc_raw', 'fs', 'dcf',...
            'chan_asag_rej')
        
    end
end
toc;
end

