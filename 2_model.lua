
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

	elseif opt.model == 'convnet_sad' then 

		-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
		model:add(nn.SpatialConvolution(nfeats, nstates[1], filtsize, filtsize))
		model:add(nn.Tanh())
		model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
		model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

		-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
		model:add(nn.SpatialConvolution(nstates[1], nstates[2], filtsize, filtsize))
		model:add(nn.Tanh())
		model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
		model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

		-- stage 3 : standard 2-layer neural network
		model:add(nn.Reshape(nstates[2]*poolsize*poolsize))
		model:add(nn.Linear(nstates[2]*poolsize*poolsize, nstates[3]))
		model:add(nn.Tanh())
		model:add(nn.Linear(nstates[3], noutputs))  

	elseif opt.model == 'convnet_frustrated' then 

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
		model:add(nn.Reshape(nstates[2]*poolsize*poolsize))
		model:add(nn.Linear(nstates[2]*poolsize*poolsize, nstates[3]))
		model:add(nn.Tanh())
		model:add(nn.Linear(nstates[3], noutputs)) 


	elseif opt.model == 'convnet_simon' then 

		 --model = nn.Sequential()
		--model:add(nn.Reshape(ninputs))
		--model:add(nn.Linear(ninputs,noutputs))
	 
		--1x20x20
		model:add(nn.SpatialConvolution(1, 15, 5, 5))
		--20x16x16
		model:add(nn.Tanh())
		model:add(nn.SpatialMaxPooling(4, 4, 1, 1))
	       --20x13x13
	       -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
	       --model:add(nn.SpatialConvolutionMap(nn.tables.random(16, 128, 4), 5, 5))
		--model:add(nn.Tanh())
	       --model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	       -- stage 3 : standard 2-layer neural network
	       model:add(nn.Reshape(15*13*13))
	       model:add(nn.Linear(15*13*13, 50))
	       model:add(nn.Tanh())
	       model:add(nn.Linear(50,noutputs))
	       model:add(nn.LogSoftMax())
             criterion = nn.ClassNLLCriterion()

	elseif  opt.model == 'convnet_other' then 

                -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization  
		model:add(nn.SpatialContrastiveNormalization(1, image.gaussian1D(5)))
                model:add(nn.SpatialConvolution(1, 8, 5, 5))
                model:add(nn.Tanh())
                model:add(nn.SpatialMaxPooling(4, 4, 4, 4))
		-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
                model:add(nn.SpatialConvolutionMap(nn.tables.random(8, 64, 4), 4, 4))
                model:add(nn.Tanh())
		-- stage 3 : standard 2-layer neural network
                model:add(nn.Reshape(64))
                model:add(nn.Linear(64,4))

	end
--elseif  opt.mode == 'test' then 
  --    model = torch.load(opt.network)
end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

 
