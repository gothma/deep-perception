require 'image'
require 'torch'
require 'lib/kitti_tools'
require 'xlua'

cmd = torch.CmdLine()
cmd:option('-size', 7480, 'number of images loaded')
cmd:option('-class', 'multi', 'which classes should be extracted (multi | binary )')
cmd:option('-simpleExtraction', false, 'if none patches should be found without knowing if they contain an object')
cmd:option('-length', 32, 'side length of one patch')
cmd:option('-crop', 'bars', 'type of image cropping')
cmd:option('-images', 'data/images/training/image_2', 'folder containing kitti images')
cmd:option('-labels', 'data/labels/label_2', 'folder containing kitti labels')
cmd:option('-dontcare', false, 'include DontCare labels')
cmd:option('-save', 'extracted_data_yuv.t7', 'the file where the images should be saved')

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

local patch_w = opt.length
local patch_h = opt.length
local totalPatches = 80256
if opt.class == 'binary' then
  -- In the binary case, there is at most for each total patch a none patch
  totalPatches = totalPatches * 2
end

trainData = {
  data = torch.DoubleTensor(totalPatches,3,patch_w,patch_h),
  labels = torch.LongStorage(totalPatches):fill(0), 
  occluded = torch.LongStorage(totalPatches):fill(0)
}
local resize = {
  bars=function (imgSub) 
    yDiff = imgSub:size(2)
    xDiff = imgSub:size(3) 
    if xDiff == yDiff then
      return image.scale(imgSub, patch_w, patch_h)
    end
    local emptyImgSub = torch.DoubleTensor(imgSub:size(1), math.max(xDiff,yDiff), math.max(xDiff,yDiff)):fill(0.5)
    if yDiff > xDiff then
      emptyImgSub[{{},{1,yDiff},{((yDiff-xDiff) / 2) + 1 ,(yDiff + xDiff) /2}}] = imgSub
    elseif xDiff > yDiff then
      emptyImgSub[{{},{((xDiff-yDiff) / 2) +1 ,(xDiff+yDiff)/2 },{1,xDiff}}] = imgSub
    end
    return image.scale(emptyImgSub,patch_w,patch_h)
  end,

  scale=function (imgSub)
    return image.scale(imgSub, patch_w, patch_h)
  end
}

function isInside(imgX, imgY, x1, y1, x2, y2)
  return x1 >= 1 and y1 >= 1 and x2 <= imgX and y2 <= imgY
end


local cntDt = 0
print('Read images')


local binaryClass = opt.class == 'binary'

for i =0, opt.size do
  xlua.progress(i, opt.size)
  local img = read_image(opt.images , i)
  local lbltbl = read_labels(opt.labels,i)
  for j = 1,table.getn(lbltbl) do
    -- Skip DontCare
    if opt.dontcare or _typeTable[lbltbl[j].type] ~= 9 then
      if (lbltbl[j].x2 > lbltbl[j].x1) and (lbltbl[j].y2 > lbltbl[j].y1) then
        if (lbltbl[j].x2 < img:size(3)) and (lbltbl[j].x1 < img:size(3)) and 
          (lbltbl[j].y2 < img:size(2)) and ( lbltbl[j].y1 < img:size(2)) then
          -- TODO: Does this take 0-based index into account?
          local imgSub = image.crop(img,lbltbl[j].x1,lbltbl[j].y1,lbltbl[j].x2,lbltbl[j].y2)
          -- Add new resize function into the resize table and call them via the parameters
          local emptyImgSub = resize[opt.crop](imgSub)
          --emptyImgSub = image.rgb2yuv(emptyImgSub)

          if binaryClass then imgSlbl = 1
          else imgSlbl = _typeTable[lbltbl[j].type] end

          cntDt = cntDt + 1
          trainData.data[cntDt] = emptyImgSub 
          trainData.labels[cntDt] = imgSlbl
          trainData.occluded[cntDt] = lbltbl[j].occluded

          if binaryClass then
            local imgSize = {x=img:size(3), y=img:size(2)}
            local patchPos = {
              x1=lbltbl[j].x1,
              y1=lbltbl[j].y1,
              x2=lbltbl[j].x2,
              y2=lbltbl[j].y2
            }

            local leftHalf = imgSize.x > patchPos.x2 + patchPos.x1
            local upperHalf = imgSize.y > patchPos.y2 + patchPos.x1

            local smallerSide = math.max(patchPos.x2 - patchPos.x1, patchPos.y2 - patchPos.y1)

            local noneImage = torch.DoubleTensor()
            --print(leftHalf, upperHalf)
            if leftHalf then
              if upperHalf then
                -- Image is mostly in left upper quater
                noneImage = image.crop(img, patchPos.x2, patchPos.y2, math.min(imgSize.x, patchPos.x2 + smallerSide), math.min(imgSize.y, (patchPos.y2 + smallerSide)))
              else
                -- Image is mostly in left lower quater
                noneImage = image.crop(img, patchPos.x2, math.max(0, patchPos.y1-smallerSide), math.min(imgSize.x, patchPos.x2 + smallerSide), patchPos.y1)
              end
            else
              if upperHalf then
                -- Image is mostly in right upper quater
                noneImage = image.crop(img, math.max(0, patchPos.x1-smallerSide), patchPos.y2, patchPos.x1, math.min(imgSize.y, patchPos.y2 + smallerSide))
              else
                -- Image is mostly in right lower quater
                noneImage = image.crop(img, math.max(0, patchPos.x1-smallerSide), math.max(0, patchPos.y1-smallerSide), patchPos.x1, patchPos.y1)
              end
            end
            cntDt = cntDt + 1
            trainData.data[cntDt] = resize[opt.crop](noneImage)
            trainData.labels[cntDt] = 2 -- none class
            trainData.occluded[cntDt] = 3 -- unknown occlution
            --print('success')
          end
        end
      end
    end
  end
end

s = trainData.data:size()
s[1] = cntDt

saveData = {
  data = torch.DoubleTensor(s),
  labels = torch.ByteTensor(cntDt),
  occluded = torch.ByteTensor(cntDt)
}

print('Copy vector to save space')
for i = 1, cntDt do
  xlua.progress(i, cntDt)
  saveData.data[i] = trainData.data[i]
  saveData.labels[i] = trainData.labels[i]
  saveData.occluded[i] = trainData.occluded[i]   
end

torch.save(opt.save,saveData)
