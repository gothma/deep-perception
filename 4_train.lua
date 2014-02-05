 
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

 
	 
 -- classes
 classes = {'1','2','3','4','5','6','7','8','9'}
 if opt.model == 'convnet_binary' then
 	classes = {'1', '2'}
 end

 -- This matrix records the current confusion across classes
 confusion = optim.ConfusionMatrix(classes)

 -- Log results to files
 trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
 testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

 -- Retrieve parameters and gradients:
 -- this extracts and flattens all the trainable parameters of the mode
 -- into a 1-dim vector
 if model then
 parameters,gradParameters = model:getParameters()
 end

 ----------------------------------------------------------------------
 print '==> configuring optimizer'

 if opt.optimization == 'CG' then
  optimState = {
  maxIter = opt.maxIter
  }
 optimMethod = optim.cg

 elseif opt.optimization == 'LBFGS' then
	optimState = {
	learningRate = opt.learningRate,
	 maxIter = opt.maxIter,
	 nCorrection = 4
 }
 optimMethod = optim.lbfgs

 elseif opt.optimization == 'SGD' then
	optimState = {
	learningRate = opt.learningRate,
	weightDecay = opt.weightDecay,
	momentum = opt.momentum,
	learningRateDecay = 1e-7
 }
 optimMethod = optim.sgd

 elseif opt.optimization == 'ASGD' then
	optimState = {
	eta0 = opt.learningRate,
	t0 = trsize * opt.t0
 }
 optimMethod = optim.asgd

 else
    error('unknown optimization method')
 end

 ----------------------------------------------------------------------
 print '==> defining training procedure'
 current_loss = 0
 function train()

	   -- epoch tracker
	   epoch = epoch or 1
	   print(epoch )
           current_loss = 0
	   -- local vars
	   local time = sys.clock()

	   -- shuffle at each epoch
           
	   shuffle = torch.randperm(trsize)

	   -- do one epoch
	   print('==> doing epoch on training data:')
	   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	   for t = 1,trainData:size(),opt.batchSize do
	      -- disp progress
	      xlua.progress(t, trainData:size())

	      -- create mini batch
	      local inputs = {}
	      local targets = {}
	      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
		 -- load new sample
		 local input = trainData.data[shuffle[i]]
		 local target = trainData.labels[shuffle[i]]
		 input = input:double()		 
		 table.insert(inputs, input)
		 table.insert(targets, target)
	      end

	      -- create closure to evaluate f(X) and df/dX
	      local feval = function(x)
			       -- get new parameters
			       if x ~= parameters then
				  parameters:copy(x)
			       end

			       -- reset gradients
			       gradParameters:zero()

			       -- f is the average of all criterions
			       local f = 0
			       
                               -- evaluate function for complete mini batch
			       for i = 1,#inputs do
				  -- estimate f
                                --  print(inputs:size())
                                --  print(inputs[1]:size()) 
				  local output = model:forward(inputs[i])
                                    
				  local err = criterion:forward(output, targets[i])
				  f = f + err

				  -- estimate df/dW
				  local df_do = criterion:backward(output, targets[i])
				  model:backward(inputs[i], df_do)

				  -- update confusion                                             
				  confusion:add(output, targets[i])
			       end

			       -- normalize gradients and f(X)
			       gradParameters:div(#inputs)
			       f = f/#inputs

			       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
        _,fs= optimMethod(feval, parameters, optimState)         
        current_loss = current_loss + fs[1]
      end
   end
   
   current_loss = current_loss / (trainData:size() )
   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(opt.save, 'model_'..opt.model..'.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end
