clear all

% FigHandle = figure(3);
% set(FigHandle, 'Position', [100, 100, 1049, 895]);
% set(FigHandle, 'Color',[1 1 1]);

sf_files = dir(fullfile('./', 'sf.*.data'));
hb_files = dir(fullfile('./', 'hb.*.data'));
% sf = 'oceQnet ','oceQsw  ','TFLUX   ',
% hb = 'TOTTTEND','ADVx_TH ','ADVy_TH ','ADVr_TH ','DFxE_TH ','DFyE_TH ','DFrE_TH ','DFrI_TH ','KPPg_TH ','WTHMASS ',
% 
% TOTTTEND(i,j,k) / 86400 = 
% 
%    - [ (ADVx_TH(i+1,j,k) – ADVx_TH(I,j,k))/CV +
% 
%    (ADVy_TH(I,j+1,k) – ADVy_TH(I,j,k))/CV +
% 
%    (ADVr_TH(I,j,k) – ADVr_TH(I,j,k+1))/CV ]
% 
%    - [ (DFxE_TH(i+1,j,k) – DFxE_TH(I,j,k))/CV +
% 
%     (DFyE_TH(I,j+1,k) – DFyE_TH(I,j,k))/CV +
% 
%    (DFrE_TH(I,j,k) – DFrE_TH(I,j,k+1))/CV ]
% 
%    -(DFrI_TH(I,j,k) – DFrI_TH(I,j,k+1))/CV ]
% 
% 	- (KPPg_TH(I,j,k) – KPPg_TH(I,j,k+1))/CV
% 
% (+ SURFACE LAYER ONLY)
% 
%  + (TFLUX-oceQsw)/(rhoConst*Cp*(RF(2)-RF(1))*hFacC(i,j,1))
% + oceQsw/(rhoConst*Cp)/((RF(2)-RF(1))*hFacC(i,j,1))*(swfrac(1)-swfrac(2))
% - (WTHMASS(i,j,1) - TsurfCor)/((RF(2)-RF(1))*hFacC(i,j,1))
% 
% 
% TsurfCor = SUM( WTHMASS(i,j,1)*RAC(i,j) ) /  globalArea
% CV = RAC(I,j) * (RF(K+1)-RF(K))  * hFacC(I,j,k)

nx = 1500;
ny = 480;
nz = 96;

spatial = nx*ny*nz;
h_spatial = nx*ny;


level = 1;
levelp = level + 1;

for ind = length(sf_files)-1

fid = fopen('RAC.data', 'r');
data = fread(fid, 'b', 'single');
fclose(fid);
RAC = reshape(data, nx,ny);  

fid = fopen('RF.data');
RF = fread(fid, 'single', 'b');
fclose(fid);
 
layer = 1;
m = memmapfile('hFacC.data', 'Format', 'single');
data = m.Data((layer-1)*spatial + (level-1)*h_spatial + 1 : (layer-1)*spatial + level*h_spatial);
hFacC = swapbytes(reshape(data, nx, ny));
    
m = memmapfile(hb_files(ind).name, 'Format', 'single');

layer = 1;
data = m.Data((layer-1)*spatial + (level-1)*h_spatial + 1 : (layer-1)*spatial + level*h_spatial);
TOTTTEND = swapbytes(reshape(data, nx, ny));

layer = 2;
data = m.Data((layer-1)*spatial + (level-1)*h_spatial + 1 : (layer-1)*spatial + level*h_spatial);
ADVx_TH = swapbytes(reshape(data, nx, ny));

layer = 3;
data = m.Data((layer-1)*spatial + (level-1)*h_spatial + 1 : (layer-1)*spatial + level*h_spatial);
ADVy_TH = swapbytes(reshape(data, nx, ny));

layer = 4;
data = m.Data((layer-1)*spatial + (level-1)*h_spatial + 1 : (layer-1)*spatial + level*h_spatial);
ADVr_TH = swapbytes(reshape(data, nx, ny));
data = m.Data((layer-1)*spatial + (levelp-1)*h_spatial + 1 : (layer-1)*spatial + levelp*h_spatial);
ADVr_THp = swapbytes(reshape(data, nx, ny));


layer = 5;
data = m.Data((layer-1)*spatial + (level-1)*h_spatial + 1 : (layer-1)*spatial + level*h_spatial);
DFxE_TH = swapbytes(reshape(data, nx, ny));

layer = 6;
data = m.Data((layer-1)*spatial + (level-1)*h_spatial + 1 : (layer-1)*spatial + level*h_spatial);
DFyE_TH = swapbytes(reshape(data, nx, ny));

layer = 7;
data = m.Data((layer-1)*spatial + (level-1)*h_spatial + 1 : (layer-1)*spatial + level*h_spatial);
DFrE_TH = swapbytes(reshape(data, nx, ny));
data = m.Data((layer-1)*spatial + (levelp-1)*h_spatial + 1 : (layer-1)*spatial + levelp*h_spatial);
DFrE_THp = swapbytes(reshape(data, nx, ny));

layer = 8;
data = m.Data((layer-1)*spatial + (level-1)*h_spatial + 1 : (layer-1)*spatial + level*h_spatial);
DFrI_TH = swapbytes(reshape(data, nx, ny));
data = m.Data((layer-1)*spatial + (levelp-1)*h_spatial + 1 : (layer-1)*spatial + levelp*h_spatial);
DFrI_THp = swapbytes(reshape(data, nx, ny));

layer = 9;
data = m.Data((layer-1)*spatial + (level-1)*h_spatial + 1 : (layer-1)*spatial + level*h_spatial);
KPPg_TH = swapbytes(reshape(data, nx, ny));
data = m.Data((layer-1)*spatial + (levelp-1)*h_spatial + 1 : (layer-1)*spatial + levelp*h_spatial);
KPPg_THp = swapbytes(reshape(data, nx, ny));

layer = 10;
data = m.Data((layer-1)*spatial + (level-1)*h_spatial + 1 : (layer-1)*spatial + level*h_spatial);
WTHMASS = swapbytes(reshape(data, nx, ny));


if level == 1
m = memmapfile(sf_files(ind).name, 'Format', 'single');

layer = 1;
data = m.Data((layer-1)*h_spatial + 1 : layer*h_spatial);
oceQnet = swapbytes(reshape(data, nx, ny));

layer = 2;
data = m.Data((layer-1)*h_spatial + 1 : layer*h_spatial);
oceQsw = swapbytes(reshape(data, nx, ny));

layer = 3;
data = m.Data((layer-1)*h_spatial + 1 : layer*h_spatial);
TFLUX = swapbytes(reshape(data, nx, ny));
end

%%%%%%%%%%% Make budget
% Test
CV = RAC(2:end-1,2:end-1) * (RF(levelp) - RF(level)) .* hFacC(2:end-1,2:end-1);  % Cell volume
surf_mass = WTHMASS(2:end-1,2:end-1) .* RAC(2:end-1,2:end-1);
global_area = 2.196468634481708E+13;  % Found as "globalArea" in STDOUT.0000
rhoConst = 1035;   % Found as "rhoConst" in STDOUT.0000
Cp = 3994; % Found as "HeatCapacity_Cp" in ~/MITgcm/model/src/set_defaults.F
TsurfCor = sum( surf_mass(:) ) / global_area;
depth = RF(1);
depthp = RF(2);
swfrac = 0.62 * exp(depth/0.6) + (1.0 - 0.62) * exp(depth/20.0);
swfracp = 0.62 * exp(depthp/0.6) + (1.0 - 0.62) * exp(depthp/20.0);


LHS = TOTTTEND(2:end-1,2:end-1) / 86400;

ADVx = -(ADVx_TH(3:end, 2:end-1) - ADVx_TH(2:end-1, 2:end-1)) ./ CV;
ADVy = -(ADVy_TH(2:end-1, 3:end) - ADVy_TH(2:end-1, 2:end-1)) ./ CV;
ADVr = -(ADVr_TH(2:end-1, 2:end-1) - ADVr_THp(2:end-1, 2:end-1)) ./ CV;
DFxE = -(DFxE_TH(3:end, 2:end-1) - DFxE_TH(2:end-1, 2:end-1)) ./ CV;
DFyE = -(DFyE_TH(2:end-1, 3:end) - DFyE_TH(2:end-1, 2:end-1)) ./ CV;
DFrE = -(DFrE_TH(2:end-1, 2:end-1) - DFrE_THp(2:end-1, 2:end-1)) ./ CV;
DFrI = -(DFrI_TH(2:end-1, 2:end-1) - DFrI_THp(2:end-1, 2:end-1)) ./ CV;
KPPg = -(KPPg_TH(2:end-1, 2:end-1) - KPPg_THp(2:end-1, 2:end-1)) ./ CV;

ADV = ADVx + ADVy + ADVr;
DIFF = DFxE + DFyE + DFrE + DFrI + KPPg;

if level == 1
surf_layer1 = (TFLUX(2:end-1,2:end-1) - oceQsw(2:end-1,2:end-1)) ./ (rhoConst*Cp*(RF(2)-RF(1))*hFacC(2:end-1,2:end-1));
surf_layer2 = oceQsw(2:end-1,2:end-1)./(rhoConst*Cp)./((RF(2)-RF(1))*hFacC(2:end-1,2:end-1))*(swfrac-swfracp);
surf_layer3 = -(WTHMASS(2:end-1,2:end-1) - TsurfCor)./((RF(2)-RF(1))*hFacC(2:end-1,2:end-1));
SL = surf_layer1 + surf_layer2 + surf_layer3;
end


RHS = ADVx + ADVy + ADVr + DFxE + DFyE + DFrE + DFrI + KPPg;
if level == 1
    RHS = RHS + surf_layer1 + surf_layer2 + surf_layer3;
end

end