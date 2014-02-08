require 'image'
require 'torch'
require 'lib/kitti_tools'
require 'lib/non_maxima'
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
function _extractPatches(img, imgIndx, testScaleSize, patchSize, strideSize) 
  local cnt = 1   
  local hw = (img:size(2) / strideSize) * (img:size(3) / strideSize)
  local tmData = torch.DoubleTensor(hw, 3, testScaleSize, testScaleSize):fill(-10) 
  local tmLoc = torch.DoubleTensor(hw, 8):fill(0)
    
  for i = 1, img:size(2), strideSize do 
    for j = 1, img:size(3), strideSize do 
      if (i + patchSize - 1 < img:size(2)) and (j + patchSize - 1 < img:size(3)) then
        tmData[cnt] = image.scale(img[{{},{i , i + patchSize - 1},{j , j + patchSize - 1}}], testScaleSize, testScaleSize)
        tmLoc[cnt] = torch.DoubleTensor({i, i + patchSize - 1, j, j + patchSize - 1, 0, 0, 0, imgIndx})  
        cnt = cnt + 1  
      end 
    end
  end  
  local tmResult = {data = torch.DoubleTensor(cnt - 1,3, testScaleSize, testScaleSize),
             locations = torch.Tensor(cnt - 1, 8)}
  for i = 1, cnt - 1 do 
    tmResult.locations[i] = tmLoc[i]
    --print(tmResult.locations[i])
    tmResult.data[i] = tmData[i]
  end 
  return tmResult
end

----------------------------------------------------------------------------
-- this is a function for normalization of the dataset
function _normalizeTestData(testData, mean, std, channels)
  testData.data = testData.data:float()
--  print '==> preprocessing data: normalize each feature (channel) globally'
  for i,channel in ipairs(channels) do 
    xlua.progress(i, testData.data:size(1))
    testData.data[{ {},i,{},{} }]:add(-mean[i])
    testData.data[{ {},i,{},{} }]:div(std[i])
  end
  --print '==> preprocessing data: normalize all three channels locally'
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
  --print("testing ====> binary classifier")
  for l = 1,testData.data:size(1) do
     -- disp progress
     
     xlua.progress(l,testData.data:size(1))
     -- get new sample
     local input = testData.data[l]
     input =input:double()  
     -- test sample
     local pred = binaryModel:forward(input)
     local tmMax, tmIndx = torch.max(pred,  1)
     testData.locations[l][5] = tmIndx[1]
     testData.locations[l][6] = tmMax[1] 
     if tmMax[1] < threshValue then 
       testData.locations[l][7] = 1
     end
  end
  return testData
end
----------------------------------------------------------------------------
-- this is the function for convolutional classification testing the patches and also threshold them
function _testConvnetClassifier(testData, threshValue, convnetModel)
  --print("testing ====> convolutional classifier")
  for l = 1,testData.data:size(1) do
    if (testData.locations[l][7] == 0) then
      -- disp progress
      
      xlua.progress(l,testData.data:size(1))
      -- get new sample
      local input = testData.data[l]
      input =input:double()  
      -- test sample
      local pred = convnetModel:forward(input)
      local tmMax, tmIndx = torch.max(pred, 1)
      testData.locations[l][5] = tmIndx[1]
      testData.locations[l][6] = tmMax[1] 
      if tmMax[1] < threshValue then 
        testData.locations[l][7] = 1
      end
    end
  end
  return testData
end
---------------------------------- create objects for nonMaximum supression------
function _createObjectsTable(locations, imgIndx)
  local _typeTable = {
    'Car',
    'Van',
    'Tram',
    'Cyclist',
    'Pedestrian',
    'Person_sitting',
    'Misc',
    'Truck',
    'DontCare'
  }
 local objects = {}
  for i = 1, locations:size(1) do
    if (locations[i][7] == 0) then
      local object = {imgNum = -1, x1 = -1, x2 = -1, y1 = -1, y2 = -1, score = -1, type = -1, threshold = 0}
      object.imgNum = locations[i][8] 
      object.y1 = locations[i][1]
      object.y2 = locations[i][2]
      object.x1 = locations[i][3]
      object.x2 = locations[i][4]
      object.type = _typeTable[locations[i][5]]
      object.score = locations[i][6]
      object.threshold = locations[i][7]
      table.insert(objects, object)
    end
  end
  if table.getn(objects) > 0 then  
    objectsNew = nonmaxima_suppression(objects)
    write_labels('results/test', imgIndx, objectsNew)
  end
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
dofile 'lib/kitti_tools.lua'
binaryModel = torch.load(opt.binaryModel)
convnetModel = torch.load(opt.convnetModel)
binaryThresh = opt.binaryThresh
convnetThresh = opt.convnetThresh
-----
-- this is function for computing the location number for all patch size of all images
local allLocCnt = _computeTensorSize(opt.indxS, opt.indxE, opt.initPatchSize
                                   , opt.patchFactor, opt.strideFactor, opt.imgFilePath)
--- ofter testing an image only store the neccessary information
--[1] = patch index number
--[2][8] = image index 
--[2][1:4] patch location
--[2][5] predicted label
--[2][6] prediction value 
--[2][7] ? 1 = thresholded(omitted) , 0 = not thresholded
local testData = {locations = torch.Tensor(allLocCnt, 7)}
                  --data = torch.DoubleTensor(opt.indxE - opt.indxS + 1, 1, 3, patch_w, patch_h),
local locIndx = 1 -- the index to latest extracted location orverall 
local mean = torch.load(opt.mean)
local std = torch.load(opt.std)
print('Read images')
for imgIndx = opt.indxS, opt.indxE do 
  
  -- this is function for computing the location number for all patch size of all images
  local allLocCnt = _computeTensorSize(imgIndx, imgIndx, opt.initPatchSize
                                   , opt.patchFactor, opt.strideFactor, opt.imgFilePath)
  local testData = {locations = torch.Tensor(allLocCnt, 8)}

  local locIndx = 0 -- the index to latest extracted location orverall 
  
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
    -- print(tostring(patchSize) .. " * " .. tostring(patchSize))
    local strideSize = patchSize * opt.strideFactor
 
    local tmTestData =  _extractPatches(img, imgIndx, opt.testScaleSize, patchSize, strideSize) ---extract all patches 
 
    tmTestData = _normalizeTestData(tmTestData, mean, std, channels)
  
    if binaryThresh ~= nil then
      tmTestData = _testBinaryClassifier(tmTestData, binaryThresh, binaryModel)
    end 

    if convnetThresh ~= nil then
      tmTestData = _testConvnetClassifier(tmTestData, convnetThresh, convnetModel)
    end 
 
    for i = 1 , tmTestData.locations:size(1)  do
      testData.locations[i + locIndx] = tmTestData.locations[i]---copy  testing results
    end  
    locIndx = locIndx + tmTestData.locations:size(1)  
  end --- end while
  _createObjectsTable(testData.locations, imgIndx)
end -- end main for loop
--print('kitti_test' .. tostring(opt.indxS) .. '_' .. tostring(opt.indxE) ..'_' .. tostring(patchSize) .. '.t7')
--torch.save('kitti_test' .. '_' .. tostring(opt.indxS) .. '_' .. tostring(opt.indxE) .. 't7', testData)
