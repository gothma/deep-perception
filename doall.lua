 
----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Loss Function')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- data:
cmd:option('-size', 'full', 'how many samples do we load: small | full | extra')
-- model:
cmd:option('-model', 'convnet_car', 'type of model : convnet_happy | convnet_sad | convnet_winking | convnet_frustrated')
-- loss:
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-trainfile', 'data/kitti_extended.t7', 'Where the trainfile lies')
cmd:option('-testfile', 'data/kitti_valid.t7',  'Where the testfile lies')
cmd:option('-extractfile', 'data/extracted_data_yuv.t7', 'Where the extracted data lies')
cmd:option('-epoches', 200, 'the number of the epoches we need to do')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-network', 'learned_model', 'learned model file name "*.net"')
cmd:option('-mode', 'train', ' the operation mode type : train | test | crossval')
cmd:option('-fold', 0, 'fold which is used for testing')
cmd:option('-folds', 0, 'if set it will do k fold cross validation')
cmd:option('-trainThreshold', 1e-3, ' the threshold value for error dicreasing')
cmd:text('-k',10,'set numbero of folding in cross validation deafult is 10')
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

if opt.mode == 'crossval' and opt.fold >= opt.folds then
  print 'The fold selected for validation needs \
  to be in the number of total folds'
  os.exit()
end

----------------------------------------------------------------------
print '==> executing all'

if opt.mode == 'crossval' then
  dofile '0_cross_data.lua'
end

if opt.mode == 'train' or opt.mode == 'crossval' then

  dofile '1_data.lua'
  dofile '2_model.lua'
  dofile '3_loss.lua'
  dofile '4_train.lua'
  dofile '5_test.lua'

----------------------------------------------------------------------
  print '==> training!'

  previous_loss = -10
 -- current_loss = 0
  kfold = 0
  for i=1,opt.epoches  ,1 do
  -- while opt.trainThreshold>=torch.abs(previous_loss-current_loss) do 
     train()

     print('epoch = ' .. epoch .. 
       ' of ' .. opt.epoches .. 
       ' current loss = ' .. current_loss ..
       ' previous loss = ' .. previous_loss)
     previous_loss = current_loss  
     test()
   end

elseif opt.mode == 'test'  then
  
  dofile '1_data.lua'
  dofile '2_model.lua'
  dofile '5_test.lua'
  test()

end
