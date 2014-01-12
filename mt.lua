
require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
noutputs = 8
nfeats = 3
width = 50
height = 50
ninputs = nfeats*width*height

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

-- hidden units, filter sizes (for ConvNet only):
nstates = {64,64,128}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)

---------------------------------------------------------------------- 
   model1 = nn.Sequential()
   model2 = nn.Sequential()
   model3 = nn.Sequential()

   im = torch.DoubleTensor(3,50,50):fill(200);
   model1:add(nn.SpatialConvolution(nfeats, nstates[1], filtsize, filtsize))
   model1:add(nn.Tanh())
   model1:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
   model1:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

	-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model2:add(nn.SpatialConvolution(nstates[1], nstates[2], filtsize, filtsize))
   model2:add(nn.Tanh())
   model2:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
   model2:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

	-- stage 3 : standard 2-layer neural network
   model3:add(nn.Reshape(nstates[2]*poolsize*poolsize))
   model3:add(nn.Linear(nstates[2]*poolsize*poolsize, nstates[3]))
   model3:add(nn.Tanh())
   model3:add(nn.Linear(nstates[3], noutputs)) 

   
   --model1:add(nn.SpatialContrastiveNormalization(1, image.gaussian1D(5)))
  -- model1:add(nn.SpatialConvolution(1, 8, 5, 5))
  -- model1:add(nn.Tanh())
  -- model1:add(nn.SpatialMaxPooling(4, 4, 4, 4))
  -- model2:add(nn.SpatialConvolutionMap(nn.tables.random(8, 64, 4), 4, 4))
  -- model2:add(nn.Tanh())
  -- model3:add(nn.Reshape(64))
  -- model3:add(nn.Linear(64,4))

   im = model1:forward(im)
   print(" model1 result")
   print(im:size())

   im = model2:forward(im);
   print(" model2 result")
   print(im:size())

   im = model3:forward(im)
   print(" model3 result")
   print(im:size())

