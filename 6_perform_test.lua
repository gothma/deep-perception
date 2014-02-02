require 'image'
require 'torch'
require 'lib/kitti_tools'
require 'xlua'
require 'nn'
--------------------- all fuctions ----------------------------------------------------
---- compute the total number of the extracted images for all images 
function _computeTensorSize(indxS, indxE, initPatchSize, patchFactor, strideFactor, imgFilePath)
  ps = 0
  cnt = 0
  local img = read_image(imgFilePath, indxS)
  local imgMinS = img:size(2)
  if (imgMinS > img:size(3)) then 
    imgMinS = img:size(3)
  end 
  while (ps * patchFactor < imgMinS) do 
    ps = ps * patchFactor
    if ps == 0 then
      ps = initPatchSize
    end
    st = ps * strideFactor
    for i = 1, img:size(2), st do 
      for j = 1, img:size(3), st do 
        if (i + ps - 1 < img:size(2)) and (j + ps - 1 < img:size(3)) then
          cnt = cnt + 1
        end
      end 
    end
  end 
  cnt = cnt * (indxE - indxS + 1)
  return cnt
end

----------------------------------------------------------------------------
---- extract all patches from an image with given patch size and stride size
function _extractPatches(img, testScaleSize, patchSize, strideSize) 
  local cnt = 1   
  local hw = (img:size(2) / strideSize) * (img:size(3) / strideSize)
  local tmData = torch.DoubleTensor(hw, 3, testScaleSize, testScaleSize):fill(-10) 
  local tmLoc = torch.DoubleTensor(hw, 7):fill(0)
    
  for i = 1, img:size(2), strideSize do 
    for j = 1, img:size(3), strideSize do 
      if (i + patchSize - 1 < img:size(2)) and (j + patchSize - 1 < img:size(3)) then
        tmData[cnt] = image.scale(img[{{},{i , i + patchSize- 1},{j , j + patchSize - 1}}], testScaleSize, testScaleSize)
        tmLoc[cnt] = torch.DoubleTensor({i, i + patchSize - 1, j, j + patchSize - 1, 0,0,0})
        print(tmLoc[cnt]) 
        cnt = cnt + 1  
      end 
    end
  end  
  local tmResult = {data = torch.DoubleTensor(cnt,3, testScaleSize, testScaleSize),
             locations = torch.Tensor(cnt, 7)}
  for i = 1, cnt do 
    tmResult.locations[i] = tmLoc[i]
    print(tmResult.locations[i])
    tmResult.data[i] = tmData[i]
  end 
  return tmResult
end

----------------------------------------------------------------------------
-- this is a function for normalization of the dataset
function _normalizeTestData(testData, mean, std, channels)
  testData.data = testData.data:float()
  print '==> preprocessing data: normalize each feature (channel) globally'
  for i,channel in ipairs(channels) do 
    xlua.progress(i, testData.data:size(1))
    testData.data[{ {},i,{},{} }]:add(-mean[i])
    testData.data[{ {},i,{},{} }]:div(std[i])
  end
  print '==> preprocessing data: normalize all three channels locally'
  local neighborhood = image.gaussian1D(13)
  local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
  for c in ipairs(channels) do
    for i = 1,testData.data:size(1) do
      xlua.progress(i, testData.data:size(1))
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
    end
  end
  return testData
end
----------------------------------------------------------------------------
-- this is the function for binary classification testing the patches and also threshold them
function _testBinaryClassifier(testData, threshValue, binaryModel)
  for l = 1,testData.data:size(1) do
     -- disp progress
     print("testing ====> binary classifier")
     xlua.progress(l,testData.data:size(1))
     -- get new sample
     local input = testData.data[l]
     input =input:double()  
     -- test sample
     local pred = binaryModel:forward(input)
     local tmMax = torch.max(pred)
     local tmIndx =0
     for im=1,pred:size(1) do 
       if tMmax == pred[im] then
         tmIndx = im  
         break
       end
     end
     testData.locations[l][5] = tmIndx
     testData.locations[l][6] = tmMax 
     if tmMax < threshValue then 
       testData.locations[l][7] = 1
     end
  end
  return testData
end
----------------------------------------------------------------------------
-- this is the function for convolutional classification testing the patches and also threshold them
function _testConvnetClassifier(testData, threshValue, convnetModel)
  for l = 1,testData.data:size(1) do
    if (testData.locations[l][7] == 0) then
      -- disp progress
      print("testing ====> convolutional classifier")
      xlua.progress(l,testData.data:size(1))
      -- get new sample
      local input = testData.data[l]
      input =input:double()  
      -- test sample
      local pred = binaryModel:forward(input)
      local tmMax = torch.max(pred)
      local tmIndx =0
      for im=1,pred:size(1) do 
        if tMmax == pred[im] then
          tmIndx = im  
          break
        end
      end
      testData.locations[l][5] = tmIndx
      testData.locations[l][6] = tmMax 
      if tmMax < threshValue then 
        testData.locations[l][7] = 1
      end
    end
  end
  return testData
end

----------------------------------------------------------------------------
--[=[cmd = torch.CmdLine()
-- test mode data setting 
cmd:option('-binaryModel', nil, 'path to binary classifier model file')
cmd:option('-convnetModel', nil, ' path to convolutional classifier model file')
cmd:option('-binaryThresh', -0.4, 'threshold for binary classifier')
cmd:option('-convnetThresh', -0.2, 'threshold for convolutional classifier')
cmd:option('-indxS', 1, 'the start index for loading image')
cmd:option('-indxE', 5, 'the end index for loading image')
cmd:option('-patchFactor', 1.3, 'the factor for increasing the patch size')
cmd:option('-strideFactor', 0.15, 'stride factor for increasing stride size for sliding patches')
cmd:option('-mean', 0, ' mean of train images should be tensor')
cmd:option('-std', 1, 'std of train images should be tensor')
cmd:option('-initPatchSize', 32, ' initialize size for patches')
cmd:option('-testScaleSize', 32, 'the scale number which needed for testing the image')
cmd:option('-imgFilePath', 'data/images/testing/image_2',' path for loading test images  [ default = data/images/resting/image_2]')
opt = opt or cmd:parse(arg or {})
]=]--
---- load models 
binaryModel = torch.load(opt.binaryModel)
convnetModel = torch.load(opt.convnetModel)
binaryThresh = opt.binaryThresh
convnetThresh = opt.convnetThresh
-----
-- this is function for computing the location number for all patch size of all images
local allLocCnt = _computeTensorSize(opt.indxS, opt.indxE, opt.initPatchSize
                                   , opt.patchFactor, opt.strideFactor, opt.imgFilePath)
--- ofter testing an image only store the neccessary information
--[1] = image index 
--[2][1:4] patch location
--[2][5] predicted label
--[2][6] prediction value 
--[2][7] ? 1 = thresholded(omitted) , 0 = not thresholded
local testData = {locations = torch.Tensor(allLocCnt, 7)}
                  --data = torch.DoubleTensor(opt.indxE - opt.indxS + 1, 1, 3, patch_w, patch_h),
local locIndx = 1 -- the index to latest extracted location orverall 
local mean = torch.load(opt.mean)
local std = torch.load(opt.std)

for imgIndx = opt.indxS, opt.indxE do 

  print('Read images')
  xlua.progress(imgIndx, opt.indxE - opt.indxS +1)
  local img = read_image(opt.imgFilePath, imgIndx)
  local imgMinSize = img:size(2)
  if imgMinSize > img:size(3) then 
    imgMinSize = img:size(3)
  end
  local patchSize = 0 
  local channels = {'y','u','v'}

  while (patchSize * opt.patchFactor < imgMinSize) do
    patchSize = patchSize * opt.patchFactor
    if (patchSize == 0) then  
      patchSize = opt.initPatchSize
    end
    print(tostring(patchSize) .. " * " .. tostring(patchSize))
    local strideSize = patchSize * opt.strideFactor
 
    local tmTestData =  _extractPatches(img, opt.testScaleSize, patchSize, strideSize) ---extract all patches 
 
    tmTestData = _normalizeTestData(tmTestData, mean, std, channels)
  
    if binaryThresh ~= nil then
      tmTestData = _testBinaryClassifier(tmTestData, binaryThresh, binaryModel)
    end 

    if convnetThresh ~= nil then
      tmTestData = _testConvnetClassifier(tmTestData, convnetThresh, convnetModel)
    end 
 
    for i = 1, tmTestData.locations:size(1) do 
      testData.locations[i + locIndx] = tmTestData.locations[i]---copy  testing results
    end 
  end --- end while
end -- end main for loop
print('kitti_test' .. tostring(opt.indxS) .. '_' .. tostring(opt.indxE) ..'_' .. tostring(patchSize) .. '.t7')
torch.save('kitti_test' .. '_' .. tostring(opt.indxS) .. '_' .. tostring(opt.indxE) ..'_' .. tostring(patchSize) .. '.t7', testData)
