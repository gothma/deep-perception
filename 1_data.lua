---------------------------------------------------------------------

require 'torch' 
require 'image'  
require 'nn'
require 'xlua'

if  opt.mode == 'train' then
----------------------------------------------------------------------
  print '==> loading dataset'
  train_file = opt.trainfile 
  test_file = opt.testfile 

  local loaded = torch.load(train_file)
  trainData = {
     data = loaded.data:double(),--:transpose(3,4),
     labels = loaded.labels,--[1],
     size = function() return trsize end
  }

  loaded = torch.load(test_file)
  testData = {
       data = loaded.data,--:transpose(3,4),
       labels = loaded.labels,--[1],
       size = function() return tesize end
  }
end
   
if opt.mode == 'train' or opt.mode == 'crossval' then   

  ---------------------------------------------------------------
  if opt.size == 'full' then
     trsize = trainData.data:size(1)
     tesize = testData.data:size(1) 
  elseif opt.size == 'small' then
     trsize = 999
     tesize = 300
  end 
  channels = {'y','u','v'}
  mean = {}
  std = {}
  ---------------------------------------------------------------------
  print '==> preprocessing data'

  trainData.data = trainData.data:float()
  testData.data = testData.data:float()

  print '==> preprocessing data: normalize each feature (channel) globally'
       
  for i,channel in ipairs(channels) do
    mean[i] = trainData.data[{ {},i,{},{} }]:mean()
    std[i] = trainData.data[{ {},i,{},{} }]:std()
    trainData.data[{ {},i,{},{} }]:add(-mean[i])
    trainData.data[{ {},i,{},{} }]:div(std[i])
  end
  torch.save('tmp_mean.t7', mean);
  torch.save('tmp_std.t7', std);
  for i,channel in ipairs(channels) do 
    testData.data[{ {},i,{},{} }]:add(-mean[i])
    testData.data[{ {},i,{},{} }]:div(std[i])
  end

  print '==> preprocessing data: normalize all three channels locally'

  neighborhood = image.gaussian1D(13)
  normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

  for c in ipairs(channels) do
    for i = 1,trainData.data:size(1) do
     xlua.progress(i, trainData.data:size(1))
     trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
    end
    for i = 1,testData.data:size(1) do
      xlua.progress(i, testData.data:size(1))
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
    end
  end

  trainMean = trainData.data[{ {},1 }]:mean()
  trainStd = trainData.data[{ {},1 }]:std()

  testMean = testData.data[{ {},1 }]:mean()
  testStd = testData.data[{ {},1 }]:std()
end
--[=[       
elseif opt.mode == 'test' then
  
  local loaded = torch.load(test_file)
  local w_patch = opt.patchSize
  local h_patch = opt.patchSize
  local cntTotal = 0 
  l_tbl = table.getn(loaded)
  testData = {data = torch.DoubleTensor(l_tbl,1,1,20,20),
            -- labels = torch.DoubleTensor(l_tbl , 1),
             size = function() return tesize end,
             locations = torch.DoubleTensor(l_tbl,1,5)}
  print(table.getn(loaded)) 
  for t = 1, table.getn(loaded) do 
   local ts= loaded[t][1]
   local cnt = 0
   local cntTotal = 0   
   local hw= ts:size(1)*ts:size(2)
   local tmData = torch.DoubleTensor(hw,20,20):fill(-10) 
   local tmLoc = torch.DoubleTensor(hw,5):fill(0)
   --print(tmData:size())
   tmDCnt =0 
   for i=1,ts:size(1) do 
     for j=1,ts:size(2) do 
       tmDCnt = tmDCnt +1; 
       if (i + w_patch - 1 < ts:size(1)) and ( j + h_patch - 1 < ts:size(2)) then
         ---print(i .. " " .. i + w_patch -1 ..  " " .. j .. " " .. j + h_patch - 1)
         tmData[tmDCnt] = image.scale(ts[{{i , i + w_patch -1},{j , j + h_patch - 1}}], 20 ,20)
         tmLoc[tmDCnt] = torch.DoubleTensor({0,i,i+w_patch-1,j,j+h_patch-1}) 
         cnt = cnt + 1         
       end 
     end
   end  
   local n = testData.data:size(1)
   cntTotal = cntTotal + cnt
   print(' cnttotal ' .. cntTotal)
   testData.data:resize(l_tbl,n + cnt,1,20,20)
    --testData.labels:resize(n + cnt):fill(0)
    testData.locations:resize(l_tbl, n + cnt,5)
    indx = 0
    if n ==1 then 
      indx = -1
    end
    for i=1,tmData:size(1) do
      if tmData[i]:max() > 0 then
        indx = indx + 1  
        testData.data[t][n + indx][1] = tmData[i]
        testData.locations[t][n + indx]=tmLoc[i]
      end
    end 
  end 
  tesize = l_tbl;

  local mean = testData.data[{{}, {},1,{},{} }]:mean()
  local std = testData.data[{{}, {},1,{},{} }]:std()
  testData.data[{{}, {},1,{},{} }]:add(-mean)
  testData.data[{{}, {},1,{},{} }]:div(std)
  testData.data = testData.data:float() 
  print '==> preprocessing data: normalize all three channels locally'
  neighborhood = image.gaussian1D(13)
  normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
  for i = 1,testData.data:size(1) do
    for j = 1,testData.data:size(1) do
      testData.data[{ i,j,{},{},{} }] = normalization:forward(testData.data[{ i,j,{},{},{} }])
    end
  end



  print("test mode data size") 
  print(testData:size())
end
]=]--
