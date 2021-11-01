% Matlab/Octave scipt file to plot iso_surface of learned energy

clear all
close all
clc

nPC = 7 % parameter to change
method = 'pca'
fname = ['results/RBC_r28L_T100R100_' method num2str(nPC) '_ons_pot'];
hrange = load([fname '_meta.txt'])
system(['gunzip ' fname '.txt.gz'])
D = load([fname '.txt']);
system(['gzip ' fname '.txt'])
[m, n] = size(D);
D = reshape(D, n, n, n);
D = permute(D, [2, 1, 3]);
%D = D(:, end:-1:1, :);
x=linspace(hrange(1,1), hrange(2,1), n);
y=linspace(hrange(1,2), hrange(2,2), n);
z=linspace(hrange(1,3), hrange(2,3), n);
[X,Y,Z]=meshgrid(x,y,z);

pot_max = max(D(:))
pot_min = min(D(:))

%isovalues = [0.002, 0.02, 0.1, 0.4] *(pot_max-pot_min) + pot_min
isovalues = [0.0025, 0.04, 0.16, 0.36] *(pot_max-pot_min) + pot_min

h = figure('Position', [100, 100, 450, 400]);
isovalue = isovalues(1);
surf1 = isosurface(X,Y,Z,D,isovalue);
p1 = patch(surf1);
isonormals(x,y,z,D,p1);
set(p1,'FaceColor','red','EdgeColor','none','FaceAlpha',0.8); 
daspect([1,1,1])
% view(-10, 30);
if nPC == 3 && strcmp(method, 'pca')
    % view(30, 45);
    view(-30, 25);
else
    view(-30, 25);
end
%view(75, 30);
camlight; 
lighting gouraud


isovalue = isovalues(2);
surf2=isosurface(x,y,z,D,isovalue);
p2 = patch(surf2);
isonormals(x,y,z,D,p2);
set(p2,'FaceColor','blue','EdgeColor','none','FaceAlpha',0.5);

isovalue = isovalues(3);
surf3=isosurface(x,y,z,D,isovalue);
p3 = patch(surf3);
isonormals(x,y,z,D,p3);
set(p3,'FaceColor','cyan','EdgeColor','none','FaceAlpha',0.3);

%if nPC==3
    isovalue = isovalues(4);
    surf4=isosurface(x,y,z,D,isovalue);
    p4 = patch(surf4);
    isonormals(x,y,z,D,p4);
    set(p4,'FaceColor','yellow','EdgeColor','none','FaceAlpha',0.15);
%end

xlabel('h_1');
ylabel('h_2');
zlabel('h_3');
xlim([hrange(1,1), hrange(2,1)]);
ylim([hrange(1,2), hrange(2,2)]);
zlim([hrange(1,3), hrange(2,3)]);

% COMMENT: manual adjust location of Legend then rerun following
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h, '-dpdf', '-r200', '-bestfit', [fname '.pdf']);

