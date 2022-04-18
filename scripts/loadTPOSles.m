function a = loadTPOSles(fnme,rithresh)
sponge_amp=0.000001;
mldthresh=0.00015; % buoyancy
for it = 1:length(fnme)
    if it == 1
        a{it}.f=0;
    elseif it == 2
        a{it}.f=2.*7.27E-5.*sind(3);
    elseif it == 3
        a{it}.f=2.*7.27E-5.*sind(1.5);        
    end
    a{it}.rho0=ncread(fnme{it},'rho0');
    a{it}.alpha=ncread(fnme{it},'alpha');
    a{it}.beta=ncread(fnme{it},'beta');
    a{it}.S0=ncread(fnme{it},'S0');
    a{it}.T0=ncread(fnme{it},'T0');
    
    a{it}.time=ncread(fnme{it},'time')./86400+datenum(1985,10,2,6,0,0);
    a{it}.z=ncread(fnme{it},'z');
    a{it}.zg=repmat(a{it}.z,[1 length(a{it}.time)]);
    a{it}.dz=a{it}.z(3)-a{it}.z(2);
    
    a{it}.tempme=ncread(fnme{it},'tempme');
    a{it}.saltme=ncread(fnme{it},'saltme');
    a{it}.ume=ncread(fnme{it},'ume');
    a{it}.vme=ncread(fnme{it},'vme');
    a{it}.wrms=ncread(fnme{it},'wrms');
    a{it}.tke=0.5.*(ncread(fnme{it},'wrms').^2+ncread(fnme{it},'vrms').^2+ncread(fnme{it},'urms').^2);
    
    
    a{it}.umeminusroms=-ncread(fnme{it},'dUdtRESTORE')./sponge_amp;
    a{it}.vmeminusroms=-ncread(fnme{it},'dVdtRESTORE')./sponge_amp;
    a{it}.tempmeminusroms=-ncread(fnme{it},'dTdtRESTORE')./sponge_amp;
    a{it}.dBdtFORCE=ncread(fnme{it},'dTdtFORCE').*9.81.*a{it}.alpha;
    a{it}.dTdtFORCE=ncread(fnme{it},'dTdtFORCE');
    a{it}.dUdtFORCE=ncread(fnme{it},'dUdtFORCE');
    a{it}.dVdtFORCE=ncread(fnme{it},'dVdtFORCE');
    a{it}.dTdtRESTORE=ncread(fnme{it},'dTdtRESTORE');
    a{it}.dUdtRESTORE=ncread(fnme{it},'dUdtRESTORE');
    a{it}.dVdtRESTORE=ncread(fnme{it},'dVdtRESTORE');
    
    
    
    a{it}.S2=ncread(fnme{it},'S2');
    a{it}.N2=ncread(fnme{it},'N2');
    a{it}.RIG=ncread(fnme{it},'RIG');
    
    a{it}.tempw=ncread(fnme{it},'tempw');
    a{it}.saltw=ncread(fnme{it},'saltw');
    a{it}.wb=9.81.*(a{it}.alpha.*a{it}.tempw+a{it}.beta.*a{it}.saltw);
    a{it}.rho=a{it}.rho0.*(1-a{it}.alpha.*(a{it}.tempme-a{it}.T0)-a{it}.beta.*(a{it}.saltme-a{it}.S0));
    a{it}.buoy=-9.81.*a{it}.rho./a{it}.rho0;
    mask=ones(size(a{it}.buoy));
    mask(2:end-1,:)=(a{it}.buoy(2:end-1,:)<repmat(mean(a{it}.buoy(end-20:end-1,:),1),[size(a{it}.buoy,1)-2 1])-mldthresh);
    zg=a{it}.zg;
    zg(logical(mask))=nan;
    a{it}.mld=nanmin(zg,[],1);
    a{it}.kappadtdz=ncread(fnme{it},'kappadtdz');
    a{it}.kappadsdz=ncread(fnme{it},'kappadsdz');
    a{it}.kappadbdz=9.81.*(a{it}.alpha.*a{it}.kappadtdz+a{it}.beta.*a{it}.kappadsdz);
    
    a{it}.kappadtdztop=ncread(fnme{it},'kappadtdztop');
    a{it}.kappadsdztop=ncread(fnme{it},'kappadsdztop');
    a{it}.kappadbdztop=9.81.*(a{it}.alpha.*a{it}.kappadtdztop+a{it}.beta.*a{it}.kappadsdztop);
    
    a{it}.dTdtSOLAR=ncread(fnme{it},'dTdtSOLAR');
    a{it}.dBdtSOLAR=9.81.*a{it}.alpha.*a{it}.dTdtSOLAR;
    a{it}.dBdtsolarsum=(sum(a{it}.dBdtSOLAR(2:end-1,:),1).*a{it}.dz)';
    a{it}.dTdtsolarsum=(sum(a{it}.dTdtSOLAR(2:end-1,:),1).*a{it}.dz)';
    
    
    a{it}.nududz=ncread(fnme{it},'nududz');
    a{it}.nudvdz=ncread(fnme{it},'nudvdz');
    
    a{it}.nududztop=ncread(fnme{it},'nududztop');
    a{it}.nudvdztop=ncread(fnme{it},'nudvdztop');
    
    
    a{it}.uw=ncread(fnme{it},'uw');
    a{it}.vw=ncread(fnme{it},'vw');
    % positive means downwards transport of momentum
    a{it}.Fim=a{it}.nududz-a{it}.uw + 1i.*(a{it}.nudvdz-a{it}.vw);
    a{it}.Fimtop=a{it}.nududztop + 1i.*(a{it}.nudvdztop);

    a{it}.Fb=a{it}.kappadbdz-a{it}.wb;
    a{it}.FT=a{it}.kappadtdz-a{it}.tempw;
    a{it}.FS=a{it}.kappadsdz-a{it}.saltw;
    
    a{it}.dudz=zeros(size(a{it}.ume));
    a{it}.dvdz=zeros(size(a{it}.vme));
    a{it}.dudz(2:end-1,:)=(a{it}.ume(3:end,:)-a{it}.ume(1:end-2,:))./(2.*a{it}.dz);
    a{it}.dvdz(2:end-1,:)=(a{it}.vme(3:end,:)-a{it}.vme(1:end-2,:))./(2.*a{it}.dz);
    a{it}.dudzim=a{it}.dudz+1i.*a{it}.dvdz;
    % maybe this could be better:
    a{it}.dudzim(1,:)=a{it}.dudzim(2,:);
    a{it}.dudzim(end,:)=a{it}.dudzim(end-1,:);
    
    a{it}.dTdz=zeros(size(a{it}.ume));
    a{it}.dSdz=zeros(size(a{it}.vme));
    a{it}.dTdz(2:end-1,:)=(a{it}.tempme(3:end,:)-a{it}.tempme(1:end-2,:))./(2.*a{it}.dz);
    a{it}.dSdz(2:end-1,:)=(a{it}.saltme(3:end,:)-a{it}.saltme(1:end-2,:))./(2.*a{it}.dz);
    % maybe this could be better:
    a{it}.dTdz(1,:)=a{it}.dTdz(2,:);
    a{it}.dTdz(end-1,:)=a{it}.dTdz(end-2,:);
    a{it}.dTdz(end,:)=a{it}.dTdz(end-1,:);
    a{it}.dSdz(1,:)=a{it}.dSdz(2,:);
    a{it}.dSdz(end-1,:)=a{it}.dSdz(end-2,:);
    a{it}.dSdz(end,:)=a{it}.dSdz(end-1,:);
    
    a{it}.KM=real(a{it}.Fim.*conj(a{it}.dudzim))./(a{it}.dudzim.*conj(a{it}.dudzim));
    a{it}.Kb=(a{it}.Fb.*a{it}.N2)./((a{it}.N2).^2);
    a{it}.KT=(a{it}.FT.*a{it}.dTdz)./(a{it}.dTdz.^2);
    a{it}.KS=(a{it}.FS.*a{it}.dSdz)./(a{it}.dSdz.^2);
    a{it}.SHEARPROD=real(a{it}.Fim.*conj(a{it}.dudzim));
    a{it}.epsilon=ncread(fnme{it},'epsilon'); 
    a{it}.epsilon(a{it}.epsilon>1)=nan;
    a{it}.epsilon(a{it}.epsilon<0)=nan;
    
end

