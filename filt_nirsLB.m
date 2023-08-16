function dodNf=filt_nirsLB(dodN,fs)
        filt_cutoff=[0.2,0.01];
        filt_order=[3,3];
        filt_type={'low','high'};
        dodNf=dodN;
        for iFilt=1:length(filt_cutoff)
            [fb,fa] = MakeFilter_lsc(1,filt_order(iFilt),fs,filt_cutoff(iFilt),filt_type{iFilt});
            dodNf=filtfilt(fb,fa,dodNf);
            dodNf=filtfilt(fb,fa,dodNf); 
        end
        cutoff=0.1;Wn=cutoff/(0.5*fs);
        [fb,fa] = butter(3,[0.9*Wn 1.1*Wn],'stop'); 
        dodNf=filtfilt(fb,fa,dodNf);
        dodNf=filtfilt(fb,fa,dodNf); 
end