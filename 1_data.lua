---------------------------------------------------------------------

require 'torch' 
require 'image'  
require 'nn'     

----------------------------------------------------------------------

train_file = opt.trainfile 
test_file = opt.testfile 


if  opt.mode == 'train' then
----------------------------------------------------------------------
	print '==> loading dataset'

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
         --[=[
        -----------------------------------------------
        --print("extend dataset") 
        ---local tmdatanum = trainData.data:size(1)
       -- local indx =0
       -- local ff= torch.Tensor(2,20,20):fill(0)
       -- local dataTen = torch.DoubleTensor(trainData.data:size(1) * 26 ,trainData.data:size(2),
                                   trainData.data:size(3),trainData.data:size(3))	
       -- local labelTen = torch.Tensor(trainData.data:size(1) * 26)	
        
       -- local secondDim = trainData.data[1][1]:size(2)
       -- local img = trainData.data[1][1]
       --- local imgDst = img
       -- local imgLbl = trainData.labels[1]
       -- local imgM = image.hflip(img)--mirror image
       -- local imgMDst = imgM
       -- for i= 1, tmdatanum do 
         -- img = trainData.data[i][1]
          --imgDst = img
          --imgM = image.hflip(img)
          --imgMDst = imgM
          --imgLbl = trainData.labels[i]
                        
         -- add original image
          --dataTen[i + indx][1] = imgDst
          --labelTen[i + indx] = imgLbl
          --indx = indx + 1 -- 1

          --add mirroed image  
          --dataTen[i + indx][1] = imgMDst
          --labelTen[i + indx] = imgLbl
          indx = indx + 1 --2
         
          --bilinear interpolaion input image
          imgDst = image.translate(img,1,1)
          imgDst[{{},1}]=torch.max(img)
          imgDst[{1,{}}]=torch.max(img)
          dataTen[i + indx][1] = imgDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--3

          imgMDst = image.translate(imgM,1,1)
          imgMDst[{{},1}]=torch.max(imgM)
          imgMDst[{1,{}}]=torch.max(imgM)
          dataTen[i + indx][1] = imgMDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--4

          imgDst = image.translate(img,-1,-1)
          imgDst[{{},20}]=torch.max(img)
          imgDst[{20,{}}]=torch.max(img)
          dataTen[i + indx][1] = imgDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--5

          imgMDst = image.translate(imgM,-1,-1)
          imgMDst[{{},20}]=torch.max(imgM)
          imgMDst[{20,{}}]=torch.max(imgM)
          dataTen[i + indx][1] = imgMDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--6

          imgDst = image.translate(img,2,2)
          imgDst[{{},{1,2}}]=torch.max(img)
          imgDst[{{1,2},{}}]=torch.max(img)
          dataTen[i + indx][1] = imgDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--7

          imgMDst = image.translate(imgM,2,2)
          imgMDst[{{},{1,2}}]=torch.max(imgM)
          imgMDst[{{1,2},{}}]=torch.max(imgM)
          dataTen[i + indx][1] = imgMDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--8

          imgDst = image.translate(img,-2,-2)
          imgDst[{{},{19,20}}]=torch.max(img)
          imgDst[{{19,20},{}}]=torch.max(img)
          dataTen[i + indx][1] = imgDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--9

          imgMDst = image.translate(imgM,-2,-2)
          imgMDst[{{},{19,20}}]=torch.max(imgM)
          imgMDst[{{19,20},{}}]=torch.max(imgM)
          dataTen[i + indx][1] = imgMDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--10

          ----------------------------
          --rotate input image 
          imgDst = image.rotate(img,0.05)
          dataTen[i + indx][1] = imgDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--11

          imgMDst = image.rotate(imgM,0.05)
          dataTen[i + indx][1] = imgMDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--12

          imgDst = image.rotate(img,-0.05)
          dataTen[i + indx][1] = imgDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--13

          imgMDst = image.rotate(imgM,-0.05)
          dataTen[i + indx][1] = imgMDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--14
        
          imgDst = image.rotate(img,0.1)
          dataTen[i + indx][1] = imgDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--15

          imgMDst = image.rotate(imgM,0.1)
          dataTen[i + indx][1] = imgMDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--16

          imgDst = image.rotate(img,-0.1)
          dataTen[i + indx][1] = imgDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--17

          imgMDst = image.rotate(imgM,-0.1)
          dataTen[i + indx][1] = imgMDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--18

          --warping bulge effect input image
          ff[{1,{5,15},{5,15}}]=-0.5 
          ff[{2,{5,15},{5,15}}]=0.5 
          
          imgDst = image.warp(img,ff)
          dataTen[i + indx][1] = imgDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--19
         
          imgMDst = image.warp(imgM,ff)
          dataTen[i + indx][1] = imgMDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--20
        
          ff[{1,{5,15},{5,15}}]=0.5 
          ff[{2,{5,15},{5,15}}]=-0.5
          
          imgDst = image.warp(img,ff)
          dataTen[i + indx][1] = imgDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--21
         
          imgMDst = image.warp(imgM,ff)
          dataTen[i + indx][1] = imgMDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--22

          ff[{1,{5,15},{5,15}}]=1
          ff[{2,{5,15},{5,15}}]=-1
          
          imgDst = image.warp(img,ff)
          dataTen[i + indx][1] = imgDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--23
         
          imgMDst = image.warp(imgM,ff)
          dataTen[i + indx][1] = imgMDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--24

        
          ff[{1,{5,15},{5,15}}]=-1
          ff[{2,{5,15},{5,15}}]=1
        
          imgDst = image.warp(img,ff)
          dataTen[i + indx][1] = imgDst
          labelTen[i + indx] = imgLbl
          indx = indx + 1--25
         
          imgMDst = image.warp(imgM,ff)
          dataTen[i + indx][1] = imgMDst
          labelTen[i + indx] = imgLbl
      --    indx = indx + 1--26
          
          --------------
     ---------- extending to non face samples 
          

          -----------------
          
          ------------------- 
          ---- adding negative examples 
        end
        trainData.data = dataTen--[{{2000,4000},{},{},{}}]
        trainData.labels = labelTen--[{{2000,4000}}]
	----------------------------------------------------------------------
         ]=]
        

        ---------------------------------------------------------------
	if opt.size == 'full' then
	   trsize = trainData.data:size(1)
	   tesize = testData.data:size(1) 
	elseif opt.size == 'small' then
	   trsize = 999
	   tesize = 300
	end 

	---------------------------------------------------------------------
	print '==> preprocessing data'

	trainData.data = trainData.data:float()
	testData.data = testData.data:float()

	print '==> preprocessing data: normalize each feature (channel) globally'

	mean = trainData.data[{ {},1,{},{} }]:mean()
	std = trainData.data[{ {},1,{},{} }]:std()
	trainData.data[{ {},1,{},{} }]:add(-mean)
	trainData.data[{ {},1,{},{} }]:div(std)

	testData.data[{ {},1,{},{} }]:add(-mean)
	testData.data[{ {},1,{},{} }]:div(std)

	print '==> preprocessing data: normalize all three channels locally'

	neighborhood = image.gaussian1D(13)

	normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

	for i = 1,trainData.data:size(1) do
	   trainData.data[{ i,{},{},{} }] = normalization:forward(trainData.data[{ i,{},{},{} }])
	end
	for i = 1,testData.data:size(1) do
	   testData.data[{ i,{},{},{} }] = normalization:forward(testData.data[{ i,{},{},{} }])
	end

	 

	trainMean = trainData.data[{ {},1 }]:mean()
	trainStd = trainData.data[{ {},1 }]:std()

	testMean = testData.data[{ {},1 }]:mean()
	testStd = testData.data[{ {},1 }]:std()

	--print('training data mean: ' .. trainMean)
	--print('training data standard deviation: ' .. trainStd)
	--print('test data mean: ' .. testMean)
	--print('test data standard deviation: ' .. testStd)

	----------------------------------------------------------------------
        print '==> visualizing data'

	 
    --    first256Samples_y = trainData.data[{ {1,265},1 }]
    --	image.display{image=first256Samples_y, nrow=16, legend='Some training examples '}
       
elseif opt.mode == 'test' then
 local loaded = torch.load(test_file)
 local w_patch = 50
 local h_patch = 50
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

