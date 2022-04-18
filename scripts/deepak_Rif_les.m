% download data_les.tar
clear all;
close all;
restoredefaultpath;
% download data_les.tar 
% from https://doi.org/10.6084/m9.figshare.14786109
fnme{1}='./data_les/ROMS_PSH_6HRLIN_0N140W_360x360x216_22OCT2020.nc'
% the netcdf files can be read by ncview 
% and by panoply *IF* you fix the z coordinate
% (top z level is ~inf, needs to be 0 or else it breaks panoply)

% Rif is not included in the output; it is derived
% I load the data into a matlab data structure using
% loadTPOSles.m 

a=loadTPOSles(fnme,0) % parameter 0 is irrelevant; a relic :/

% add an Rif variable to the data structure:
a{1}.Rif=a{1}.Fb./(a{1}.epsilon+a{1}.Fb); 
% note Fb includes subgrid scale temperature flux, kappadTdz, 
% resolved temperature flux -tempw, subgridscale salinity flux kappadSdz,
% and resolved salinity flux -saltw ; see loadTPOSles.m

a{1}.Reb=a{1}.epsilon./(1e-6.*a{1}.N2);


% we don't trust Rif where turbulence is weak; this is just
% parameterization
% let's just crudely remote regions of low epsilon with a fixed threshold
mask=a{1}.epsilon<1e-8;
a{1}.Rif(mask)=nan;

figure('position',[50 50 1200 700]);
subplot(2,1,1),...
pcolor(a{1}.time,a{1}.z(2:end-1),a{1}.Rif(2:end-1,:));
shading flat;
caxis([0.0 0.35])
hold on
plot(a{1}.time,a{1}.mld,'k-');
colormap(gca,jet(20));
datetick('x')
colorbar;
title('Rif')
ylabel('depth m')

subplot(2,1,2),...
contourf(a{1}.time,a{1}.z(2:end-1),real(log10(a{1}.Reb(2:end-1,:))),[0:.5:5],'linestyle','none');
shading flat;
caxis([0 5])
hold on
plot(a{1}.time,a{1}.mld,'k-');
colormap(gca,jet(10));
datetick('x')
colorbar;
title('log10 Reb')
ylabel('depth m')
xlabel('Date 1985')

% The plot shows that the LES simulates rather high Rif~0.2. 
% Why is this different from what Deepak infers from observations? 

% Some thoughts:
% First, the methods for inferring Rif and Reb are different in observations and LES and neither are perfect.
% 
% Observations are of small-scale variance of T and S, which are used to estimate chi, epsilon. 
% Then, temperature, salinity, and buoyancy fluxes are estimated using flux gradient relations.
% Conversely, LES directly simulates the dominant energy containing scales of the turbulence, 
% and of the temperature and salinity and hence buoyancy fluxes, 
% but LES uses ~1-meter scale  gradients in shear/T to parameterize dissipation scales. 
% So, you can see how Rif varies dynamically in LES (e.g. how it varies with Reb and Rig, surface forcing, S2, N2, etc.) 
% and compare with observations. But, it is not clear that you can use LES or observations to resolve  
% questions about uncertainties associated with the methodologies of each,
% with the possible exception of some aspects of the sampling uncertainties.
% You can directly address other questions about methodological uncertainties with flux tower observations or DNS. 
% On the other hand, both of these tools have limitations. 
% The DNS resolves dissipation scales but the box is too small (due to <cm scale grid cells) to fully resolve the TKE and fluxes, 
% while flux towers only work in shallow water or in the lower atmospheric
% boundary layer and may not be able to sample the parameter space relevant for turbulence in the open ocean thermocline.


%I do think practical progress might be made via at least 2 and maybe 3 angles:
%1) quantify variability of Rif in the LES as well as observations
%2) examine possible causes of variability in Rif in observations and LES. Are there similarities in these spatio-temporal patterns or parameter dependencies (e.g. relatively invariant Rif with Reb?)? Could there be an explanation due to difference in parameter space?
%I would like to be able to predict for subgrid parameterization in LES (and in ocean models), how Rif varies with Rig or Reb at 1 m.
% It seems Reb is not very useful, but maybe Rig is?
%Either way, modelling Rif is very important.
%3) I’m even less certain about this third angle, but perhaps the virtual mooring data from LES can be used to examine the consequences of spatial or ensemble averaging on the observational estimate (and you can do some similar assessment of averaging sensitivity in observations)? 
%In LES, you have subridscale viscosity and diffusivity at each model level and state variables (U,V,W,T,S), which you could use to estimate parameterized dissipation proportional to nusgs (dudz)^2 and kappasgs (dTdz)^2 with coefficient assuming isotropy and compare with the laterally averaged statistics. 
% Conversely, you could explore sensitivity to your sequence of averaging/analysis in observations, e.g. first average eps/chi, N2, dTdz, over various numbers of profiles, then run through Rif calculations. 







