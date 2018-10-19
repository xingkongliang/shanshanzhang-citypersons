clear all
ndt=0;
load('annotations/anno_val.mat');
dt = anno_val_aligned;

for i=1:length(dt)
    bbs=dt{i}.bbs;    
    for ibb=1:size(bbs,1)
        bb = bbs(ibb,:);  
        
        if (bb(1) == 1) % | (bb(1) == 5)            
            ndt=ndt+1;              
            dt_coco(ndt).image_id=i; 
            dt_coco(ndt).category_id=1;
            dt_coco(ndt).bbox=double(bb(2:5));
            dt_coco(ndt).score= 1.0;
        end
    end
end

dt_string = gason(dt_coco);
fp = fopen('evaluation/val_dt6.json','w');
fprintf(fp,'%s',dt_string);
fclose(fp);
