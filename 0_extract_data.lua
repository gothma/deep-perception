require 'image'
require 'torch'
require 'lib/kitti_tools'
require 'xlua'

cmd = torch.CmdLine()
cmd:option('-size', 7480, 'number of images loaded')
opt = opt or cmd:parse(arg or {})

-- Tables take strings as index, on false index an error is thown
_typeTable = {
  Car=1,
  Van=2,
  Tram=3,
  Cyclist=4,
  Pedestrian=5,
  Person_sitting=6,
  Misc=7,
  Truck=8,
  DontCare=9
}

local patch_w = 32
local patch_h = 32
trainData = {
  data = torch.DoubleTensor(80256,3,patch_w,patch_h),
  labels = torch.LongStorage(80265):fill(0), 
  occluded = torch.LongStorage(80265):fill(0)
}

local cntDt = 0
print('Read images')
for i =0, opt.size do
  xlua.progress(i, opt.size)
  local img = read_image('data/images/training/image_2' , i)
  local lbltbl = read_labels('data/labels/label_2',i)
  for j = 1,table.getn(lbltbl) do
    
    if (lbltbl[j].x2 > lbltbl[j].x1) and (lbltbl[j].y2 > lbltbl[j].y1) then
      if (lbltbl[j].x2 < img:size(3)) and (lbltbl[j].x1 < img:size(3)) and 
        (lbltbl[j].y2 < img:size(2)) and ( lbltbl[j].y1 < img:size(2)) then

        local imgSub = image.crop(img,lbltbl[j].x1,lbltbl[j].y1,lbltbl[j].x2,lbltbl[j].y2)
        yDiff = imgSub:size(2)
        xDiff = imgSub:size(3) 

        local emptyImgSub = torch.DoubleTensor(imgSub:size(1), math.max(xDiff,yDiff), math.max(xDiff,yDiff)):fill(1)

        if yDiff > xDiff then
           emptyImgSub = image.rgb2yuv(emptyImgSub)    
           imgSub = image.rgb2yuv(imgSub)
           emptyImgSub[{{},{1,yDiff},{((yDiff-xDiff) / 2) + 1 ,(yDiff + xDiff) /2}}] = imgSub
        elseif xDiff > yDiff then
           emptyImgSub = image.rgb2yuv(emptyImgSub)    
           imgSub = image.rgb2yuv(imgSub)
           emptyImgSub[{{},{((xDiff-yDiff) / 2) +1 ,(xDiff+yDiff)/2 },{1,xDiff}}] = imgSub
        end
        emptyImgSub = image.scale(emptyImgSub,patch_w,patch_h)
        imgSlbl = _typeTable[lbltbl[j].type]
        cntDt = cntDt + 1
        trainData.data[cntDt] = emptyImgSub 
        trainData.labels[cntDt] = imgSlbl
        trainData.occluded[cntDt] = lbltbl[j].occluded 
      end
    end
  end
end

trainData.data = trainData.data[{{1,cntDt},{},{}}]
tmoccluded = trainData.occluded
tmlabels = trainData.labels

trainData.labels = torch.LongStorage(cntDt):fill(0)
trainData.occluded = torch.LongStorage(cntDt):fill(0)

print('Write images in t7 format')
for i = 1, cntDt do
  xlua.progress(i, cntDt)
  trainData.labels[i] = tmlabels[i]
  trainData.occluded[i] = tmoccluded[i]     
end
torch.save('extracted_data_yuv.t7',trainData)
