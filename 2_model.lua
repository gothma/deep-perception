
require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
require 'optim'
 
----------------------------------------------------------------------
print '==> define parameters'

-- 4-class problem
noutputs = 9

-- input dimensions
nfeats = 3
width = 32
height = 32
ninputs = nfeats*width*height

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

-- hidden units, filter sizes (for ConvNet only):
nstates = {64,64,128}
nstatesBinary = {32,32,64}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)

----------------------------------------------------------------------
print '==> construct model'
model = nn.Sequential()

if opt.mode == 'train' or opt.mode == 'crossval' then

    if opt.model == 'convnet_car' then 

	-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
		model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
		model:add(nn.Tanh())
		model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
		model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

		-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
		model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
		model:add(nn.Tanh())
		model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
		model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

		-- stage 3 : standard 2-layer neural network
		model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
		model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
		model:add(nn.Tanh())
		model:add(nn.Linear(nstates[3], noutputs)) 

	elseif opt.model == 'convnet_binary' then

		noutputs = 2

	-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
		model:add(nn.SpatialConvolutionMM(nfeats, nstatesBinary[1], filtsize, filtsize))
		model:add(nn.Tanh())
		model:add(nn.SpatialLPPooling(nstatesBinary[1], 2, poolsize, poolsize, poolsize, poolsize))
		model:add(nn.SpatialSubtractiveNormalization(nstatesBinary[1], normkernel))

		-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
		model:add(nn.SpatialConvolutionMM(nstatesBinary[1], nstatesBinary[2], filtsize, filtsize))
		model:add(nn.Tanh())
		model:add(nn.SpatialLPPooling(nstatesBinary[2], 2, poolsize, poolsize, poolsize, poolsize))
		model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

		-- stage 3 : standard 2-layer neural network
		model:add(nn.Reshape(nstatesBinary[2] * filtsize * filtsize))
		model:add(nn.Linear(nstatesBinary[2] * filtsize * filtsize, nstates[3]))
		model:add(nn.Tanh())
		model:add(nn.Linear(nstatesBinary[3], noutputs)) 

	elseif opt.model then
		model = torch.load(opt.model)
	end
--elseif  opt.mode == 'test' then 
  --    model = torch.load(opt.network)
end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

 
