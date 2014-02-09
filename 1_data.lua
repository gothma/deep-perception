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
  if (trainData.data:size(2) == 3) then
    channels = {'y', 'u', 'v'}
  elseif (trainData.data:size(2) == 1) then
    channels = {'y'}
  end
  mean = {}
  std = {}
  ---------------------------------------------------------------------
  print '==> preprocessing data'

  trainData.data = trainData.data:float()
  testData.data = testData.data:float()

  print '==> preprocessing data: normalize each feature (channel) globally'
  for i,channel in ipairs(channels) do
    mean[i] =  trainData.data[{ {},i,{},{} }]:mean()
    std[i] =  trainData.data[{ {},i,{},{} }]:std()
    trainData.data[{ {},i,{},{} }]:add(-mean[i])
    trainData.data[{ {},i,{},{} }]:div(std[i])
  end
  torch.save(opt.mode .. '_' .. opt.model .. '_mean.t7', mean);
  torch.save(opt.mode .. '_' .. opt.model .. '_std.t7', std);
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

end

