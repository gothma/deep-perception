require 'image'
require 'torch'
require 'nn'

function _disturb(instance, chance)
  disturbed_instance = torch.Tensor():resizeAs(instance):copy(instance)
  for y=2,instance:size(1)-1 do
    for x=2,instance:size(2)-1 do
      if torch.bernoulli(chance) == 1 then
      xshift = torch.random() % 3 - 1
      yshift = torch.random() % 3 - 1
      disturbed_instance[y][x] = instance[y+yshift][x+xshift]
      end
    end
  end
  return disturbed_instance
end

print '==> loading dataset'

local loaded = torch.load('extracted_data_yuv.t7')
trainData = {
  data = loaded.data:double(),--:transpose(3,4),
  labels = loaded.labels
}
local trdbSize = trainData.data:size()

local validation_set_size = 100
validationData = {
  data = torch.DoubleTensor(validation_set_size * 9 * 2,trdbSize[2],trdbSize[3],trdbSize[4]),--:transpose(3,4),
  labels = torch.ByteTensor(validation_set_size * 9 * 2)
}

-----------------------------------------------
print("extend dataset") 
local tmdatanum = trdbSize[1]
local indx =0
local dataTen = torch.DoubleTensor((trdbSize[1] - (validation_set_size * 9)) * 2 ,trdbSize[2],
                         trdbSize[3],trdbSize[3]) 
local labelTen = torch.Tensor((trdbSize[1] - (validation_set_size * 9)) * 2)  

local img = trainData.data[1]
local imgLbl = trainData.labels[1]
local imgM = image.hflip(img)--mirror image
local hold_back_counter = torch.LongStorage(9):fill(0)
local val_cnt = 0
local i = 0

for j= 1, tmdatanum do 
  img = trainData.data[j]
  imgM = image.hflip(img)
  imgLbl = trainData.labels[j]
      
  if hold_back_counter[imgLbl] < validation_set_size  then
    -- add original image
    val_cnt  = val_cnt +1
    validationData.data[val_cnt] = img
    validationData.labels[val_cnt] = imgLbl   
    -- add mirroed image
    val_cnt  = val_cnt +1
    validationData.data[val_cnt] = imgM
    validationData.labels[val_cnt] = imgLbl   

    hold_back_counter[imgLbl] = hold_back_counter[imgLbl] + 1 
  else     
    -- add original image
      i = i + 1
    dataTen[i + indx] = img
    labelTen[i + indx] = imgLbl

    --add mirroed image  
    indx = indx + 1 -- 1
    dataTen[i + indx] = imgM
    labelTen[i + indx] = imgLbl
  end       
end
trainData.data = dataTen--[{{2000,4000},{},{},{}}]
trainData.labels = labelTen--[{{2000,4000}}]
--------------------------------------------

torch.save('kitti_extended.t7', trainData)
torch.save('kitti_valid.t7', validationData)
