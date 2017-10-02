require 'nn'
require 'gnuplot'
require 'image'
require 'optim'
require 'cunn'
require 'sys'

torch.setdefaulttensortype('torch.FloatTensor')

--coded directly for GPU!
--load data
dataPath = "/media/kswiss/ExtraDrive1/ELVESNet/merged/"
dataAll = torch.load(dataPath..'ALLDataTensor.dat')
labelAll = torch.load(dataPath..'ALLDataLocation.dat')
realDataPath = "/media/kswiss/ExtraDrive1/ELVESNet/rawdata/"
realData = torch.load(realDataPath..'ALLDataTensor.dat')
dataAll = dataAll:float()
labelAll = labelAll:float()

--normalize the data to the highest value in that trace. 
for i=1,dataAll:size(1) do
   if(torch.max(dataAll[i])~=0) then dataAll[i] = dataAll[i]/torch.max(dataAll[i]) end
end
realData = realData/torch.max(realData)


collectgarbage()

data1 = dataAll[{ {1,6153-1},{},{} }]:float()
label1 = labelAll[{ {1,6153-1},{} }]:float()
data2 = dataAll[{ {6153+1,6675-1},{},{} }]:float()
label2 = labelAll[{ {6153+1,6675-1},{} }]:float()
data3 = dataAll[{ {6675+1,dataAll:size()[1]},{},{} }]:float()
label3 = labelAll[{ {6675+1,dataAll:size()[1]},{} }]:float()

dataAll = torch.cat(data1,data2,1):cat(data3,1)
data1 = nil
data2 = nil
data3 = nil
labelAll = torch.cat(label1,label2,1):cat(label3,1)
label1 = nil
label2 = nil
label3 = nil

collectgarbage()
--9779 ELVES left to play with.


--insert a singleton in position 2 to add the dimension of the image channel to the dataset
modelSoliton = nn.Unsqueeze(2)
dataAll = modelSoliton:forward(dataAll)


print("# images: " .. dataAll:size(1))
print("# channels: " .. dataAll:size(2))
print("# pixels: " .. dataAll:size(3))
print("# time bins: " .. dataAll:size(4))

trsize = 8000
mbsize = 200

nb_minibatches = math.floor(trsize/mbsize)
print("\n" .. nb_minibatches .. " minibatches")
if (trsize/mbsize)%1 ~= 0 then print(sys.COLORS.red .. "WARNING: the number of minibatches does NOT span the whole training set!\n") end
   
tesize = 1000

learningRateValue = 1e-2
print_every = 20
nb_epoch = 200

train_data = {
   data = dataAll[{ {1,trsize},{},{},{} }],
   labels = labelAll[{ {1,trsize} }],
   size = function() return trsize end
}
train_data_batch = {
   data = torch.Tensor(mbsize, train_data.data:size(2),train_data.data:size(3),train_data.data:size(4)),
   labels = torch.Tensor(mbsize,2),
   size = function() return mbsize end
}
test_data = {
   data = dataAll[{ {trsize+1,trsize+tesize},{},{},{} }],
   labels = labelAll[{ {trsize+1,trsize+tesize} }],
   size = function() return tesize end
}
real_data = {
   data = realData,
   labels = labelAll[{ {1,1}}]:zero(),
   size = function() return 1 end
}


print(train_data:size(), " ELVES in train dataset. " .. train_data.data:type())
print(train_data_batch:size(), " ELVES in train minibatch. " .. train_data_batch.data:type())
print(test_data:size(), " ELVES in test dataset. " .. test_data.data:type())

train_data_batch.data = train_data_batch.data:cuda()
train_data_batch.labels = train_data_batch.labels:cuda()

--print(torch.max(train_data.data,3)[2][1][1][1])


local model = nn.Sequential()

model:add(nn.Reshape(1320*150))
model:add(nn.Linear(1320*150,256))
model:add(nn.Sigmoid())
--model:add(nn.Dropout(0.25))
--model:add(nn.Linear(2056,128))
--model:add(nn.Sigmoid())
--model:add(nn.Dropout(0.5))
model:add(nn.Linear(256,2))

local criterion = nn.MSECriterion()

print(sys.COLORS.red .. '\nTHE MODEL:')
print(model)
print("\n")

--send model and criterion to GPU
criterion = criterion:cuda()
model = model:cuda()

local params, grads = model:getParameters()
params:uniform(-0.01, 0.01)

--===========
-- OPTIMIZER
--===========
-- return loss, grad
local feval = function(x)
   if x ~= params then
      params:copy(x)
   end
   grads:zero()
   
   -- forward
   local outputs = model:forward(train_data_batch.data)
   local loss = criterion:forward(outputs, train_data_batch.labels)
   -- backward
   local dloss_doutput = criterion:backward(outputs, train_data_batch.labels)
   model:backward(train_data_batch.data, dloss_doutput)
   
   return loss, grads
end

--======
-- TRAIN
--======
local losses = torch.Tensor(nb_epoch,nb_minibatches):zero()
local optim_state = {learningRate = learningRateValue}
for i = 1, nb_epoch do
   --get random permutation of integers from 1 to the train data size
   local shuffle = torch.randperm(train_data:size())
   for k = 1, nb_minibatches do
      for j = 1, mbsize do
	 train_data_batch.data[j] = train_data.data[shuffle[j+(k-1)*mbsize]]
	 train_data_batch.labels[j] = train_data.labels[shuffle[j+(k-1)*mbsize]]
      end
      local _, loss = optim.sgd(feval, params, optim_state)
      losses[i][k] = loss[1] -- append the new loss
--print(i .. " " .. train_data.labels[shuffle[1+(k-1)*mbsize]][1] .. " " .. 1+(k-1)*mbsize .. " " .. loss[1])
   end
   if i % print_every == 0 then
      --print(shuffle[1] .. " " .. train_data_batch.labels[1][1] .. " " .. train_data.labels[shuffle[1]][1])
      print("iteration " .. i .. ", loss = " .. torch.mean(losses[i]))
   end
end

--classification error on another train minibatch -- not sure how to do that. will just do ((E-O)/E)^2
local shuffle = torch.randperm(train_data:size())
for j = 1, mbsize do
   train_data_batch[j] = train_data[shuffle[j]]
end

class_performancelat = torch.Tensor(train_data:size()):zero()
class_performancelon = torch.Tensor(train_data:size()):zero()
counterperformance = 0
for i=1,train_data:size() do
   local groundtruthlat = train_data.labels[i][1]
   local groundtruthlon = train_data.labels[i][2]
   local input = torch.Tensor(train_data.data[i]):cuda()
   if torch.mean(train_data.data[i]) ~= 0 then
      local prediction = model:forward(input)
      local predictionlat = prediction[1][1]
      local predictionlon = prediction[1][2]
      class_performancelat[i-counterperformance]  = torch.abs((groundtruthlat - predictionlat)/groundtruthlat)
      class_performancelon[i-counterperformance]  = torch.abs((groundtruthlon - predictionlon)/groundtruthlon)
      counterperformance = counterperformance+1
   end
   input = nil
end
print("TRAIN ERROR: ",counterperformance, torch.mean(class_performancelat), torch.mean(class_performancelon))

--======
-- TEST
--======
test_data.data = test_data.data:cuda()
test_data.labels = test_data.labels:cuda()
class_performancelat = torch.Tensor(test_data:size()):zero()
class_performancelon = torch.Tensor(test_data:size()):zero()
counterperformance = 0
for i=1,test_data:size() do
   local groundtruthlat = test_data.labels[i][1]
   local groundtruthlon = test_data.labels[i][2]
   if torch.mean(test_data.data[i]) ~= 0 then
      local prediction = model:forward(test_data.data[i])
      local predictionlat = prediction[1][1]
      local predictionlon = prediction[1][2]
      class_performancelat[i-counterperformance]  = torch.abs((groundtruthlat - predictionlat)/groundtruthlat)
      class_performancelon[i-counterperformance]  = torch.abs((groundtruthlon - predictionlon)/groundtruthlon)
      counterperformance = counterperformance+1
   end
end
print("TEST ERROR: ", counterperformance, torch.mean(class_performancelat), torch.mean(class_performancelon))


--test on real data.
print("\n Real event reconstructed latitude and longitude: ")
real_data.data = real_data.data:cuda()
real_data.labels = real_data.labels:cuda()
local prediction = model:forward(real_data.data)
local predictionlat = prediction[1][1]
local predictionlon = prediction[1][2]
print(predictionlat,predictionlon)

