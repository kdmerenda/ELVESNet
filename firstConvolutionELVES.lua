require 'nn'
require 'gnuplot'
require 'image'
require 'optim'
require 'cunn'

--load data
dataPath = "/media/kswiss/ExtraDrive1/ELVESNet/merged/"
dataAll = torch.load(dataPath..'ALLDataImage.dat')
labelAll = torch.load(dataPath..'ALLDataLocation.dat')

--insert a singleton in position 2
modelSoliton = nn.Unsqueeze(2)
dataAll = modelSoliton:forward(dataAll)
print(dataAll:size())

--9781 to play with
trsize = 7000
tesize = 1000

--gpu work
cuda = false

--net input parameters
n_inputs_1 = 1 --number of input planes to layer 1
n_filters_1 = 16 --number of convolutional filters to layer 1
n_inputs_2 = n_filters_1 --number of input planes to layer 2
n_filters_2 = 128 --number of convolutional filters to layer 2
n_convx = 2 --conv filter size - use square filter for now? image not square.
n_convy = 2 --conv filter size - use square filter for now? image not square.
n_pool = 2 --size of pooling filter
n_pool_dx = 2 --size of steps by which to move filter.

print_every = 1
nb_epoch = 5


--need to redefine the labels based on discrete grid. For first attempt, define 4 sections instead of continuous possibility of values. Here, the elves were created at random in the following range:   float lowestLON=-68, highestLON=-62;  float lowestLAT=-37, highestLAT=-32; If we discretize a possible grid in which the ELVES can be (eg 4 sections) then all the labels have to be redefined to the number of a given section (eg: 1,2,3,4). For starters, the numbering will be done wrt to the geodetic coordinates (earth cs) not wrt to the detector. I believe this will be useful for when the full detector is used. Define nSections as the number of discrete sections in which an ELVES can be.
latLow = -37
latHigh = -32
lonLow = -68
lonHigh = -62
nSectionsLat = 2
nSectionsLon = 2
nSectionsTot = nSectionsLat*nSectionsLon
--making  rectangular grid, loop through i first and j next. 
-- latLow + i*(abs(latLow-latHigh) / nSectionsLat)
-- lonLow + j*(abs(lonLow-lonHigh) / nSectionsLon)
--
-- 16 section example:                          4 section example:                            
--           LAT			                  LAT			       
--           -33 |-----|-----|-----|-----|                -33 |-----------|-----------|     
--               |     |     |     |     |                    |           |           |     
--               |  1  |  2  |  3  |  4  |                    |           |           |     
--               |     |     |     |     |                    |           |           |     
--               |-----|-----|-----|-----|                    |     1     |     2     |     
--               |     |     |     |     |                    |           |           |     
--               |  5  |  6  |  7  |  8  |                    |           |           |     
--               |     |     |     |     |                    |           |           |     
--               |-----|-----|-----|-----|                    |-----------|-----------|     
--               |     |     |     |     |                    |           |           |
--               |  9  | 10  | 11  | 12  |                    |           |           |
--               |     |     |     |     |                    |           |           |
--               |-----|-----|-----|-----|                    |     3     |     4     |
--               |     |     |     |     |                    |           |           |
--               | 13  | 14  | 15  | 16  |                    |           |           |
--               |     |     |     |     |                    |           |           |
--           -37 |-----|-----|-----|-----|                -37 |-----------|-----------|     
--              -68                     -62 LON              -68                     -62 LON

--LatLonMap = torch.Tensor(nSectionsLon+1,nSectionsLat+1)
labelAllReworked = torch.Tensor(labelAll:size()[1])
for k=1,labelAllReworked:size()[1] do
   labelAllReworked[k] = 0
   for i=1,nSectionsLat do
      bbreak = false
      for j=1,nSectionsLon do
	 latLowTmp  = latHigh - (i)*(torch.abs(latLow-latHigh) / nSectionsLat) 
	 latHighTmp = latHigh - (i-1)*(torch.abs(latLow-latHigh) / nSectionsLat)
	 lonLowTmp  = lonLow + (j-1)*(torch.abs(lonLow-lonHigh) / nSectionsLon) 
	 lonHighTmp = lonLow + (j)*(torch.abs(lonLow-lonHigh) / nSectionsLon)
	 if labelAll[k][2] >= lonLowTmp and labelAll[k][2] <= lonHighTmp and labelAll[k][1] >= latLowTmp and labelAll[k][1] <= latHighTmp then
	    labelAllReworked[k] = j+(i-1)*nSectionsLon
	    bbreak = true
	    break
	 end
      end
      if (bbreak) then break end
   end
   if labelAllReworked[k] == 0 then  print(labelAll[k],k) end 
end

--gnuplot.hist(labelAllReworked,nSectionsTot*10)   
print("Number of uncategorized ELVES: " .. torch.sum(torch.eq(labelAllReworked,0)) .. "\n\n\n")

--save the data in a struct and format the train and test sets to get part of the full dataset
train_data = {
   data = dataAll[{ {1,trsize},{},{},{} }],
   labels = labelAllReworked[{ {1,trsize} }],
   size = function() return trsize end
}
test_data = {
   data = dataAll[{ {trsize+1,trsize+tesize},{},{},{} }],
   labels = labelAllReworked[{ {trsize+1,trsize+tesize} }],
   size = function() return tesize end
}

print(train_data.data:size())
print(train_data.size(), " Location of a train ELVES (Section): ",train_data.labels[2])
print(test_data.size(), " Location of a test ELVES (Section): ",test_data.labels[2])

-- all images here are the integrated signal in time. So take one of the 9781 images and test a possible comvolution on a 'grayscale'
--imageSample = torch.Tensor(1,dataAll:size()[2],dataAll:size()[3])
--imageSample[1] = dataAll[13]
--print(imageSample)
--image.display(imageSample)--run with qlua for this...
--gnuplot.raw("set xrange restore")
--gnuplot.raw("unset border")
--gnuplot.raw("unset xtics")
--gnuplot.raw("unset ytics")
--gnuplot.raw("unset colorbox")
--gnuplot.imagesc(imageSample[1])

---------------------------
--Container:
---------------------------
--to be able to plug layers in a feed-forward fully connected manner
local model = nn.Sequential()

--===========
--FIRST LAYER
--===========
---------------------------
--Convolution Layer:
---------------------------
--A convolution is an integral that expresses the amount of overlap of one function g as it is shifted over another function f. It therefore "blends" one function with another.  Applies a 2D convolution over an input image composed of several input planes. nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]), here use default step size of 1 to move kernel. 
model:add(nn.SpatialConvolution(n_inputs_1,n_filters_1,n_convx,n_convy))

---------------------------
--Transfer Function Layer:
---------------------------
--introduce a non-linearity after a parametrized layer. This is the main computation performed by the artificial neuron. The activation function z_i = f(x,w_i) and the output function y_i = f(z_i) are summed up with the term transfer functions.

model:add(nn.Tanh()) --f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
--model:add(nn.ReLU()) --f(x) = max(0, x)
--model:add(nn.Sigmoid()) --f(x) = 1 / (1 + exp(-x))
--model:add(nn.SoftMax()) --f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)


---------------------------
--Convolution Layer:
---------------------------
--For each of the regions represented by the filter, we will take the max of that region and create a new, output matrix where each element is the max of a region in the original input. Applies 2D max-pooling operation in kWxkH regions by step size dWxdH steps. The number of output features is equal to the number of input planes.
model:add(nn.SpatialMaxPooling(n_pool,n_pool,n_pool_dx,n_pool_dx));

---------------------------
--Miscellanous Layer:
---------------------------
--During training, Dropout masks parts of the input using binary samples from a bernoulli distribution. Each input element has a probability of p of being dropped, i.e having its commensurate output element be zero. This has proven an effective technique for regularization and preventing the co-adaptation of neurons
model:add(nn.Dropout(0.50));

--============
--SECOND LAYER
--============
model:add(nn.SpatialConvolution(n_inputs_2,n_filters_2,n_convx,n_convy))
--model:add(nn.Tanh())
model:add(nn.ReLU()) 
model:add(nn.SpatialMaxPooling(n_pool,n_pool,n_pool_dx,n_pool_dx));
model:add(nn.Dropout(0.25));

--===========
--THIRD LAYER
--===========
--this is where the complex space is simplified back to a linear value. Applies a linear transformation to the incoming data, i.e. y = Ax + b. The input tensor given in forward(input) must be either a vector (1D tensor) or matrix (2D tensor)
--model:add(nn.Reshape(n_filters_2*n_convx*n_convy))
--model:add(nn.Linear(n_filters_2*n_convx*n_convy, n_filters_2*n_convx*n_convy))
model:add(nn.Reshape(128*4*14))
model:add(nn.Linear(n_filters_2*n_convx*n_convy*14, n_filters_2*n_convx*n_convy))
--model:add(nn.Tanh())
model:add(nn.ReLU()) 
model:add(nn.Dropout(0.5));
model:add(nn.Linear(n_filters_2*n_convx*n_convy, nSectionsTot))
model:add(nn.LogSoftMax())

--==============
-- LOSS FUNCTION
--==============
--local criterion = nn.ClassNLLCriterion()
local criterion = nn.CrossEntropyCriterion()

--======
-- TRAIN
--======
--taken from practical 4 of oxford class
local params, grads = model:getParameters()
--initialize weights. needed?
params:uniform(-0.01, 0.01)

--===========
--CUDA
--===========
if(cuda) then
   model:cuda()
   train_data.data = train_data.data:cuda()
   train_data.labels = train_data.labels:cuda()
   test_data.data = test_data.data:cuda()
   test_data.labels = test_data.labels:cuda()
   criterion = criterion:cuda()
end



--===========
--OPTIMIZER
--===========
-- return loss, grad
local feval = function(x)
   if x ~= params then
      params:copy(x)
   end
   grads:zero()
   
   -- forward
   local outputs = model:forward(train_data.data)
   local loss = criterion:forward(outputs, train_data.labels)
   -- backward
   local dloss_doutput = criterion:backward(outputs, train_data.labels)
   model:backward(train_data.data, dloss_doutput)
   
   return loss, grads
end

-- optimization loop
--do the training optimization over many epochs. 
local losses = {}
local optim_state = {learningRate = 1e-1}
for i = 1, nb_epoch do
   local _, loss = optim.adagrad(feval, params, optim_state)
   losses[#losses + 1] = loss[1] -- append the new loss

   if i % print_every == 0 then
      print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))
   end
end

print(params:size()) --I assume these can be saved so that the net doesn't have to be retrained.

gnuplot.figure()
gnuplot.plot({'losses',
  torch.range(1, #losses), -- x-coordinates
  torch.Tensor(losses),    -- y-coordinates
  '-'}
)


print("The NN: \n",string.format( model))

--classification error on train set
local log_probs = model:forward(train_data.data)
local _, predictions = torch.max(log_probs, 2)
print('# correct for train set:')
print(torch.mean(torch.eq(predictions:long(), train_data.labels:long()):double()))

--classification error on test set
local log_probs = model:forward(test_data.data)
local _, predictions = torch.max(log_probs, 2)
print('# correct for test set:')
print(torch.mean(torch.eq(predictions:long(), test_data.labels:long()):double()))


--res = net:forward(imageSample)
print("\n\n")
--print(res:size())
--image.display(res)


