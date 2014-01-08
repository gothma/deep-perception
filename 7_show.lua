require 'image'
require 'torch'
require 'nn'

local testData = torch.load('smiley_test.t7')
local locateData = torch.load('results/model_convnet_sad.net_location.t7')
local imgDis= torch.DoubleTensor(locateData:size(1),49,49)
local cnt =0
for l= 1, locateData:size(1) do 
 if (locateData[l][1]==1) then
  cnt = cnt +1
  print(locateData[l]) 
  imgDis[cnt]=image.crop(testData[1][1],locateData[l][4],locateData[l][2],locateData[l][5],locateData[l][3])
 end
end
print(cnt)
print(imgDis:size())
local imgDist=imgDis[{{1,cnt}, {},{}}]
imgDis = imgDist
image.display{image=imgDis, nrow= 20} 
