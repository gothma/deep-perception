
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
  -- local vars
  local time = sys.clock()

  -- averaged param use?
  if average then
    cachedparams = parameters:clone()
    parameters:copy(average)
  end

  -- test over test data
   
  print('==> testing on test set:')
 
  if opt.mode == 'train' or opt.mode == 'crossval' then
    for t = 1,testData:size() do
    -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]
      input =input:double()  
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input) 
      confusion:add(pred, target)
    end
  -- timing
  time = sys.clock() - time
  time = time / testData:size()
  print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matri
  print(confusion)

   --   update log/plot
  testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  if opt.plot then
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    testLogger:plot()
  end

  -- averaged param use?
  if average then
    --restore parameters
    parameters:copy(cachedparams)
  end
  confusion:zero()

  -- next iteration:

end
--[=[
  elseif opt.mode == 'test' then
    for t = 1,testData:size() do
      for l = 1,testData.data[t]:size(1) do
        -- disp progress
        xlua.progress(l,testData.data[t]:size(1))

        -- get new sample
        local input = testData.data[t][l]
        input =input:double()  
        --local target = testData.labels[t]

        -- test sample
        local pred = model:forward(input)
        local tmmax = torch.max(pred)
        local tmindx =0
        for im=1,pred:size(1) do 
          if tmmax == pred[im] then
            tmindx = im  
            break
          end
        end
        testData.locations[t][l][1]=tmindx 
        --print(pred) 
        --confusion:add(pred, target)
        --confusion:add(pred,1)
      end 
    end
    torch.save(opt.network .. '_location_all.t7' , testData.locations)   
  end
end
]=]--
