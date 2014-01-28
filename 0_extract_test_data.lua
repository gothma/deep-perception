require 'image'
require 'torch'
require 'lib/kitti_tools'
require 'xlua'

cmd = torch.CmdLine()
-- test mode data setting 
cmd:option('-indxS', 1, 'the start index for loading image')
cmd:option('-indxE', 5, 'the end index for loading image')
cmd:option('-patchFactor', 1.5, 'the factor for increasing the patch size')
cmd:option('-stride', 5, 'the stride for extracting patches')
cmd:option('-mean', 0, ' mean of train images')
cmd:option('-std', 1, 'std of train images')
opt = opt or cmd:parse(arg or {})

-- Tables take strings as index, on false index an error is thown
local patch_w = 0 
local patch_h = 0 
local channels = {'y','u','v'}
for i = 1, 4 do
  patch_w = patch_w * opt.patchFactor
  patch_h = patch_h * opt.patchFactor
  if ((patch_w == 0) and (patch_h == 0)) then  
    patch_w = 64
    patch_h = 64
  end
  print(tostring(patch_h) .. tostring(patch_w))
  local testData = {data = torch.DoubleTensor(opt.indxE - opt.indxS + 1, 1, 3, patch_w, patch_h),
         	    locations = torch.DoubleTensor(opt.indxE - opt.indxS +1, 1, 6)}
  for imgIndx = opt.indxS, opt.indxE do 
    print('Read images')
    xlua.progress(imgIndx, opt.indxE - opt.indxS +1)
    local img = read_image('data/images/testing/image_2', imgIndx)
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
