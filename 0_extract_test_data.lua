require 'image'
require 'torch'
require 'lib/kitti_tools'
require 'xlua'
require 'nn'

function _computeTensorSize(indxS, indxE, imgPath, initPatch, patchFact, stridFact)
  ps = 0
  cnt = 0
  local img = read_image('data/images/testing/image_2', indxS)
  local imgMinS = img:size(2)
  if (imgMinS > img:size(3)) then 
    imgMinS = img:size(3)
  end 
  while (ps * patchFact < imgMinS) do 
    ps = ps * patchFactor
    if ps == 0 then
      ps = initPatch
    end
    st = ps * stridFact
    for i = 1, img:size(2), st do 
      for j = 1, img:size(3), st do 
        cnt = cnt + 1
      end 
    end
  end 
  cnt = cnt * (indxE - indxS + 1)
  return cnt
end

cmd = torch.CmdLine()
-- test mode data setting 
cmd:option('-binaryModel', 'path to binary classifier model file')
cmd:option('-convnetModel', ' path to convolutional classifier model file')
cmd:option('-indxS', 1, 'the start index for loading image')
cmd:option('-indxE', 5, 'the end index for loading image')
cmd:option('-patchFactor', 1.3, 'the factor for increasing the patch size')
cmd:option('-strideFactor', 0.15, 'stride factor for increasing stride size for sliding patches')
cmd:option('-mean', 0, ' mean of train images should be tensor')
cmd:option('-std', 1, 'std of train images should be tensor')
cmd:option('-initPatch', 32, ' initialize size for patches')
cmd:option('-imgPath', 'data/images/testing/image_2',' path for loading test images  [ default = data/images/resting/image_2]')
opt = opt or cmd:parse(arg or {})

---- load models 
binaryModel = opt.binaryModel
convnetModel = opt.convnetModel
------
-- Tables take strings as index, on false index an error is thown
local patch_w = 0 
local patch_h = 0 
local channels = {'y','u','v'}
local cnt = _computeTensorSize(opt.indxS, opt.indxE, opt.imgPath, opt.initPatch, opt.patchFactor, opt.tridFactor)
local testData = {locations = torch.Tensor(cnt, 7)}
--data = torch.DoubleTensor(opt.indxE - opt.indxS + 1, 1, 3, patch_w, patch_h),
local locIndx = 1
for imgIndx = opt.indxS, opt.indxE do 
 
  print('Read images')
  xlua.progress(imgIndx, opt.indxE - opt.indxS +1)
  local img = read_image('data/images/testing/image_2', imgIndx)
  imgMinSize = img:size(2)
  if imgMinSize > img:size(3) then 
    imgMinSize = img:size(3)
  end
  
  while (patch_w * opt.patchFactor < imgminSize) do
    patch_w = patch_w * opt.patchFactor
    patch_h = patch_h * opt.patchFactor
    if ((patch_w == 0) and (patch_h == 0)) then  
      patch_w = opt.patchSize
      patch_h = opt.patchSize
    end
    print(tostring(patch_h) .. tostring(patch_w))
    local cnt = 1   
    local hw = (img:size(2) / opt.stride) * (img:size(3) / opt.stride)
    local tmData = torch.DoubleTensor(hw, 3, 32, 32):fill(-10) 
    local tmLoc = torch.DoubleTensor(hw, 6):fill(0)
    
    for i = 1, img:size(2), opt.stride do 
      for j = 1, img:size(3), opt.stride do 
        if (i + patch_w - 1 < img:size(2)) and (j + patch_h - 1 < img:size(3)) then
          tmData[cnt] = image.scale(img[{{},{i , i + patch_w - 1},{j , j + patch_h - 1}}], 32, 32)
          tmLoc[cnt] = torch.DoubleTensor({i, i + patch_w - 1, j, j + patch_h - 1, 0, 0}) 
          cnt = cnt + 1  
        end 
      end
    end  
    local n = testData.data[imgIndx]:size(1)
    local	 indx = 0
    if n == 1 then 
      indx = -1
    end
    print(n) 
    testData.data:resize(opt.indxE - opt.indxS, n + cnt, 3, 32, 32)
    testData.locations:resize(opt.indxE - opt.indxS, n + cnt, 6)
    for i = 1, cnt do 
      indx = indx + 1  
      testData.locations[imgIndx][n + indx] = tmLoc[i]
      testData.data[imgIndx][n + indx] = tmData[i]
    end 
  end -- end  for loop
  local mean = {}
  local std = {}
  ---------------------------------------------------------------------
  print '==> preprocessing data'
  testData.data = testData.data:float()
  for i,channel in ipairs(channels) do
    mean[i] = testData.data[{ {},i,{},{} }]:mean()
    std[i] = testData.data[{ {},i,{},{} }]:std()
    testData.data[{ {},i,{},{} }]:add(-mean[i])
    testData.data[{ {},i,{},{} }]:div(std[i])
  end
  
  torch.save('kitti_test' .. tostring(opt.indxS) .. '_' .. tostring(opt.indxE) ..'_' .. tostring(patch_h) .. '.t7', testData)
end -- end main for loop
