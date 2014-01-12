require 'image'
require 'torch'
require 'lib/kitti_tools'

function _typeToNumber(imgType)
 tmResult = 0 
 if imgType == 'Car' then
   tmResult = 1 
 end
 if imgType == 'Van' then
   tmResult = 2 
 end
 if imgType == 'Tram' then
   tmResult = 3
 end
 if imgType == 'Cyclist' then
   tmResult = 4 
 end
 if imgType == 'Pedestrian' then
   tmResult = 5 
 end
 if imgType == 'Person_sitting' then
   tmResult = 6
 end
 if imgType == 'Misc' then
   tmResult = 7
 end
 if imgType == 'DontCare' then
   tmResult = 8
 end
 return tmResult
end


local patch_w = 50
local patch_h = 50
trainData = {
  	      data = torch.DoubleTensor(80256,3,patch_w,patch_h),
   	      labels = torch.LongStorage(80265), 
              occluded = torch.LongStorage(80265)
	   }

local cntDt = 0

for i =0, 7480 do 
  local img = read_image('data/images/training/image_2' , i)
  local lbltbl = read_labels('data/labels/label_2',i)
  for j = 1,table.getn(lbltbl) do
    
    if (lbltbl[j].x2 > lbltbl[j].x1) and (lbltbl[j].y2 > lbltbl[j].y1) then
      if (lbltbl[j].x2 < img:size(3)) and (lbltbl[j].x1 < img:size(3)) and 
        (lbltbl[j].y2 < img:size(2)) and ( lbltbl[j].y1 < img:size(2)) then
        local imgSub = image.crop(img,lbltbl[j].x1,lbltbl[j].y1,lbltbl[j].x2,lbltbl[j].y2)
        imgSub = image.rgb2yuv(imgSub)
        imgSub = image.scale(imgSub,patch_w,patch_h)
        imgSlbl = _typeToNumber(lbltbl[j].type)
        cntDt = cntDt + 1
        trainData.data[cntDt] = imgSub 
        trainData.labels[cntDt] = imgSlbl
        trainData.occluded[cntDt] = lbltbl[j].occluded 
      end
    end
  end
end

trainData.data = trainData.data[{{1,cntDt},{},{}}]
trainData.labels = trainData.labels[{{1,cntDt}}]
trainData.occluded = trainData.occluded[{{1,cntDt}}]
torch.save('extracted_data_yuv.t7',trainData)
