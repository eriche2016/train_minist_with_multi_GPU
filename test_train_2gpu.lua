--[[
--Jan 18, 2016: 
--  result:only batch_size is greater than a value(here by in my machine, training with 2 gpu, batch_size > 256), does 2 gpu train faster than 1 gpu 
--  result achieved:
--              batch_size       time_to_learn_1_batch ( 2gpu ms/1gpu ms)
--              32               31/21
--              64               50/40
--              128              89/78
--              256              156/216
--              384              199/302
--]]    

--[[
--  test training a network using two gpu 
--]]

require 'nn'
require 'optim'
require 'cutorch'
require 'cunn'
require 'm_GPU_utils'
local cmd = torch.CmdLine() 
cmd:text() 
cmd:text('test a neural network which is trained on 2 gpu')
cmd:option('-gpuid', 0, 'base gpu id')
cmd:option('-nGPU', 2, 'number of gpus to be used')
cmd:option('-display_every', 10, 'how often to display the loss')
cmd:option('-batch_size', 384, 'the size of a batch')
cmd:option('-save', './', 'directory to save stuff like log files and model')
cmd:text() 

opt = cmd:parse(arg)

-- prepair data 

cutorch.setDevice(opt.gpuid+1) 

local mnist = require 'mnist'

local train_set = mnist.traindataset() 
Xt = train_set.data
Yt = train_set.label

local test_set = mnist.testdataset() 
Xv = test_set.data 
Yv = test_set.label 

Yt[Yt:eq(0)] = 10 
Yv[Yv:eq(0)] = 10 

-- construct model 
function make_model_ct() 
    local model = nn.Sequential()

    model:add(nn.Reshape(1, 28, 28))
    model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
    model:add(nn.Tanh())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
    model:add(nn.Tanh())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    model:add(nn.Reshape(64 * 4 * 4))
    model:add(nn.Linear(64 * 4 * 4, 200))
    model:add(nn.Tanh())
    model:add(nn.Linear(200, 10))
    model:add(nn.LogSoftMax())
    
    local criterion = nn.ClassNLLCriterion() 
    print("Model: ")
    print(model)
    return model, criterion  
end

model, criterion =  make_model_ct() 

-- shift model and criterion to cuda 
if opt.gpuid >= 0 then 
    model = model:cuda()
    criterion = criterion:cuda()
end 

if opt.nGPU > 1 then 
    model = makeDataParallel(model, opt.nGPU)
end 

local sgd_config = {
                    learningRate = 0.001,  
                    learningRateDecay = 0, 
                    momentum = 0, 
                   } 

local params, grad_params = model:getParameters() 

classes = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'} 
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- training procedure 
local Nt = Xt:size(1) 


function train()
    -- record current epoch
    epoch = epoch or 1 
    if opt.nGPU > 1 then 
        cutorch.synchronize() 
    end 

    model:training() 
    
    print("online epoch # " .. epoch ..  " [ batchSize = " .. opt.batch_size)
    -- start timing  
    local time = sys.clock() 
    local nbatches =  0 

    for i = 1, Nt, opt.batch_size do 
        -- extract a minibatch of size batch_size data
        nbatches = nbatches + 1 
        --[[
        if opt.nGPU > 1 then 
             cutorch.synchronize() 
        end
        --]]

        local j = math.min(i+opt.batch_size-1, Nt)
        local Xb = Xt[{{i, j}}]:cuda() 
        local Yb = Yt[{{i, j}}]:cuda() 
        
        function feval(x)
           
            collectgarbage()
            
            if x~= params then 
                params:copy(x)
            end 
            
            -- reset gradients 
            grad_params:zero() 

            local outputs = model:forward(Xb) 
            local f = criterion:forward(outputs, Yb)
            local df_do = criterion:backward(outputs, Yb)
            model:backward(Xb, df_do)

            grad_params:div(j-i+1)

            -- update confusion table 
            for m = 1, j-i+1 do 
                confusion:add(outputs[m], Yb[m])
            end 

            return f, grad_params

        end 

        optim.sgd(feval, params, sgd_config)

        if opt.nGPU > 1 then  
            --this function will copy the parameters to modules located in other gpu 
            model:syncParameters()
            -- or 
            --model:apply(function(m) if m.syncParameters then m:syncParameters() end end)
            cutorch.synchronize() 
        end  

    end 
 
    if opt.nGPU > 1 then 
        cutorch.synchronize() 
    end 

    -- time taken 
    time = sys.clock() - time

    time =  time / nbatches

    print("time to learn 1 batch of size " .. opt.batch_size .. " is " .. (time * 1000) .. " ms")
    -- print confusion matrix
    print(confusion)
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    
    confusion:zero()
    -- compute accuracy of test dataset 
    accur = compute_accuracy(Xv, Yv, model, opt.batch_size)
    epoch = epoch + 1 
end  


-- compute accuracy of test dataset 
function compute_accuracy(Xv,Yv,net,batch_size)
    model:evaluate()
    local batch_size = batch_size or 32
    local Nv = Xv:size(1)
    local lloss = 0.0

    local time = sys.clock() 
    
    for i = 1, Nv, batch_size do
        local j = math.min(i+batch_size-1,Nv)
        local Xb = Xv[{{i,j}}]:cuda()
        local Yb = Yv[{{i,j}}]:cuda()
        -- testing 
        local out = net:forward(Xb)
        -- confusion 
        for m = 1, j-i+1 do 
            confusion:add(out[m], Yb[m])
        end 

        local tmp,YYb = out:max(2)
        lloss = lloss + YYb:eq(Yb):sum()
        
    end
    -- timing 
    time = sys.clock() - time 
    time = time / Nv 

    print("time to test 1 example = " .. (time * 1000) .. "ms") 
    
    -- print confusion matrix 
    print(confusion)
    testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
    confusion:zero() 

    return (100*lloss/Nv)
end

for i = 1, 100 do 
    train() 
end 

