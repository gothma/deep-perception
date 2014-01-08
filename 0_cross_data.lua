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

	local loaded = torch.load('smiley_train.t7')
	trainData = {
	   data = loaded.data:double(),--:transpose(3,4),
	   labels = loaded.labels	}

       local validation_set_size = 25
       validationData = {
	   data = torch.DoubleTensor(validation_set_size*4,1,20,20),--:transpose(3,4),
	   labels = torch.ByteTensor(validation_set_size*4)}

        -----------------------------------------------
        print("extend dataset") 
        local tmdatanum = trainData.data:size(1)
        local indx =0
        local ff= torch.Tensor(2,20,20):fill(0)
        local dataTen = torch.DoubleTensor((trainData.data:size(1) - (validation_set_size*4)) * 28,trainData.data:size(2),
                                   trainData.data:size(3),trainData.data:size(3))	
        local labelTen = torch.Tensor((trainData.data:size(1) - (validation_set_size*4)) * 28)	
         
        local secondDim = trainData.data[1][1]:size(2)
        local img = trainData.data[1][1]
        local imgDst = img
        local imgLbl = trainData.labels[1]
        local imgM = image.hflip(img)--mirror image
        local imgMDst = imgM
        local chance = 0.05
        
       
	local hold_back_counter = 0
	local last_label = nil 
        local val_cnt = 0
        local i = 0
        for j= 1, tmdatanum do 
          img = trainData.data[j][1]
          imgDst = img
          imgM = image.hflip(img)
          imgMDst = imgM
          imgLbl = trainData.labels[j]
          

              
          if hold_back_counter < validation_set_size and last_label ~= imgLbl then
          val_cnt  = val_cnt +1
          validationData.data[val_cnt][1]=img
          validationData.labels[val_cnt]=imgLbl		
          hold_back_counter = hold_back_counter + 1 
  	  else     
                  i = i +1
		  last_label = imgLbl
		  hold_back_counter = 0
		  
		 -- add original image
		  dataTen[i + indx][1] = imgDst
		  labelTen[i + indx] = imgLbl
		  indx = indx + 1 -- 1

		  --add mirroed image  
		  dataTen[i + indx][1] = imgMDst
		  labelTen[i + indx] = imgLbl
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
		  indx = indx + 1--26
		  
		  ---------  disturb image
		  imgDst = _disturb(img,chance)
		  dataTen[i + indx][1] = imgDst
		  labelTen[i + indx] = imgLbl
		  indx = indx + 1--25
		 
		  imgMDst = _disturb(imgM,chance)
		  dataTen[i + indx][1] = imgMDst
		  labelTen[i + indx] = imgLbl
		  --indx = indx + 1--26
	       

         end		    
          --------------
     ---------- extending to non face samples 
          

          -----------------
          
          ------------------- 
          ---- adding negative examples 
        end
        trainData.data = dataTen--[{{2000,4000},{},{},{}}]
        trainData.labels = labelTen--[{{2000,4000}}]
       --------------------------------------------

        local loaded = torch.load('smiley_non6.t7')
        local tdbs = trainData.data:size(1)
        local ldbs = loaded.data:size(1)
        trainData.data:resize(tdbs+ldbs,1,20,20)
        trainData.data[{{tdbs+1,tdbs+ldbs},{1}}]=loaded.data[{{1,ldbs},{1}}]  
 
        trainData.labels:resize(tdbs+ldbs)
        for i=1,ldbs do
          trainData.labels[tdbs+i]=loaded.lables[i]
        end
       
   
        




------------------------------------------------- 
        torch.save('smiley_extended.t7', trainData)
        torch.save('smiley_valid.t7', validationData)
	----------------------------------------------------------------------
        







