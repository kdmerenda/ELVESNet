require 'nn'
require 'gnuplot'
require 'image'
require 'optim'
require 'cunn'
require 'sys'

torch.setdefaulttensortype('torch.FloatTensor')

--load real data
realDataPath = "/media/kswiss/ExtraDrive1/ELVESNet/mergeddata/out/"
realData = torch.load(realDataPath..'ALLDataTensor.dat')
realLabel = torch.load(realDataPath..'ALLDataLocation.dat')
--getting the name of the truth data file in order
realName = torch.load(realDataPath..'ALLDataName.dat')
realNameTMP = {}
for i=1,realName:size(1) do
   realNameTMPTMP = {}
   for j=1,realName:size(2) do
      realNameTMPTMP[j] = realName[i][j]
   end
   realNameTMP[i] = torch.CharStorage(realNameTMPTMP):string()
   realNameTMPTMP = nil
--   print(realNameTMP[i].." " ..realLabel[i][1].." "..realLabel[i][2])
end
realName = realNameTMP
realNameTMP = nil

--load simulated data
dataPath = "/media/kswiss/ExtraDrive1/ELVESNet/merged/"
dataAll = torch.load(dataPath..'ALLDataTensor.dat')
labelAll = torch.load(dataPath..'ALLDataLocation.dat')
dataAll = dataAll:float()
labelAll = labelAll:float()

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

dataAll = dataAll[{{1,5000},{},{}}]
labelAll = labelAll[{{1,5000}}]
originalSimDataSize = dataAll:size(1)
collectgarbage()

--9779 ELVES left to play with but had to reduce the count to allow fro multiplicity of multipage events.
--need to implement 1, 2,and 3 page events to teach the NN, for that I will set to 0 ranges in traces from 51 to 150 and from 100 to 150 and cat into the dataset, effectively creating 9779*3 elves, the labes will be the same. 
dataAll = torch.cat(dataAll,dataAll,1):cat(dataAll,1)
--dataAll = torch.cat(dataAll,dataAll,1)
labelAll = torch.cat(labelAll,labelAll,1):cat(labelAll,1)
--labelAll = torch.cat(labelAll,labelAll,1)
dataAll[{ {originalSimDataSize+1,2*originalSimDataSize},{},{51,150} }] = 0 
dataAll[{ {2*originalSimDataSize+1,3*originalSimDataSize},{},{101,150} }] = 0 
print(originalSimDataSize, dataAll:size(1))
--gnuplot.pngfigure('threepage.png')
--gnuplot.imagesc(dataAll[originalSimDataSize+2],'color')
--gnuplot.plotflush()

--scramble multipage events
local shuffle = torch.randperm(dataAll:size(1))
dataAllTMP = dataAll
labelAllTMP = labelAll
for ievt=1,dataAll:size(1) do
   dataAll[ievt] = dataAllTMP[shuffle[ievt]]
   labelAll[ievt] = labelAllTMP[shuffle[ievt]]
end
dataAllTMP = nil
labelAllTMP = nil

collectgarbage()

--normalize the data to the highest value in that trace.  do that after multipage events are created
for i=1,dataAll:size(1) do
   if(torch.max(dataAll[i])~=0) then dataAll[i] = dataAll[i]/torch.max(dataAll[i]) end
end
for i=1,realData:size(1) do
   realData[i] = realData[i]/torch.max(realData[i])
end

--what range is the simulated data in?
latmin = torch.min(labelAll[{{},1}])
latmax = torch.max(labelAll[{{},1}])
lonmin = torch.min(labelAll[{{},2}])
lonmax = torch.max(labelAll[{{},2}])
print("Latitude Range: " ..latmin.." "..latmax)
print("Longitude Range: "..lonmin.." "..lonmax)

collectgarbage()
--insert a singleton in position 2 to add the dimension of the image channel to the dataset
modelSoliton = nn.Unsqueeze(2)
dataAll = modelSoliton:forward(dataAll)
modelSoliton2 = nn.Unsqueeze(2)
realData = modelSoliton2:forward(realData)

print("# images: " .. dataAll:size(1))
print("# channels: " .. dataAll:size(2))
print("# pixels: " .. dataAll:size(3))
print("# time bins: " .. dataAll:size(4))

trsize = 14000
mbsize = 200

nb_minibatches = math.floor(trsize/mbsize)
print("\n" .. nb_minibatches .. " minibatches")
if (trsize/mbsize)%1 ~= 0 then print(sys.COLORS.red .. "WARNING: the number of minibatches does NOT span the whole training set!\n") end
   
tesize = 1000

learningRateValue = 1e-2
print_every = 10
nb_epoch = 100

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

--free RAM from data
dataAll = nil

real_data = {
   data = realData,
   labels = realLabel,
   size = function() return realData:size(1) end
}

print(train_data:size(), " ELVES in train dataset. " .. train_data.data:type())
print(train_data_batch:size(), " ELVES in train minibatch. " .. train_data_batch.data:type())
print(test_data:size(), " ELVES in test dataset. " .. test_data.data:type())
print(real_data:size(), " ELVES in real dataset. " .. real_data.data:type())

train_data_batch.data = train_data_batch.data:cuda()
train_data_batch.labels = train_data_batch.labels:cuda()

local model = nn.Sequential()

model:add(nn.Reshape(1320*150))
model:add(nn.Linear(1320*150,256))
model:add(nn.Sigmoid())
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

--clear GPU
train_data_batch.labels = nil
train_data_batch.data = nil

--classification error on another train set -- not sure how to do that. will just do MSE 1/n * ((E-O))^2
class_performancelat = torch.Tensor(train_data:size()):zero()
class_performancelon = torch.Tensor(train_data:size()):zero()
counterperformance = 0
latitudeperformance = 0
longitudeperformance = 0
for i=1,train_data:size() do
   local groundtruthlat = train_data.labels[i][1]
   local groundtruthlon = train_data.labels[i][2]
   local input = torch.Tensor(train_data.data[i]):cuda() --send one element at a time on the GPU
   if torch.mean(train_data.data[i]) ~= 0 then
      local prediction = model:forward(input)
      local predictionlat = prediction[1][1]
      local predictionlon = prediction[1][2]
      class_performancelat[i-counterperformance]  = torch.abs((groundtruthlat - predictionlat)*(groundtruthlat - predictionlat))
      class_performancelon[i-counterperformance]  = torch.abs((groundtruthlon - predictionlon)*(groundtruthlon - predictionlon))
      latitudeperformance = latitudeperformance + class_performancelat[i-counterperformance]
      longitudeperformance = longitudeperformance + class_performancelon[i-counterperformance]
      counterperformance = counterperformance+1
   end
   input = nil
end
print("TRAIN ERROR (w/o blanks): ", counterperformance, latitudeperformance/counterperformance,longitudeperformance/counterperformance)

--======
-- TEST
--======
--test data small enough to send all on gpu
test_data.data = test_data.data:cuda()
test_data.labels = test_data.labels:cuda()
class_performancelat = torch.Tensor(test_data:size()):zero()
class_performancelon = torch.Tensor(test_data:size()):zero()
counterperformance = 0
latitudeperformance = 0
longitudeperformance = 0
for i=1,test_data:size() do
   local groundtruthlat = test_data.labels[i][1]
   local groundtruthlon = test_data.labels[i][2]
   if torch.mean(test_data.data[i]) ~= 0 then
      local prediction = model:forward(test_data.data[i])
      local predictionlat = prediction[1][1]
      local predictionlon = prediction[1][2]
      class_performancelat[i-counterperformance]  = torch.abs((groundtruthlat - predictionlat)*(groundtruthlat - predictionlat))
      class_performancelon[i-counterperformance]  = torch.abs((groundtruthlon - predictionlon)*(groundtruthlon - predictionlon))
      latitudeperformance = latitudeperformance + class_performancelat[i-counterperformance]
      longitudeperformance = longitudeperformance + class_performancelon[i-counterperformance] 
      counterperformance = counterperformance+1
   end
end
print("TEST ERROR (w/o blanks): ", counterperformance, latitudeperformance/counterperformance,longitudeperformance/counterperformance)

--clear GPU
test_data.data = nil
test_data.labels = nil

--===============
-- TEST REAL DATA
--===============
-- real data small enough to go on gpu
real_data.data = real_data.data:cuda()
--real_data.labels = real_data.labels:cuda()
class_performancelat = torch.Tensor(real_data:size()):zero()
class_performancelon = torch.Tensor(real_data:size()):zero()
realLat = torch.Tensor(real_data:size()):zero()
predLat = torch.Tensor(real_data:size()):zero()
realLon = torch.Tensor(real_data:size()):zero()
predLon = torch.Tensor(real_data:size()):zero()
intrainFOV = torch.Tensor(real_data:size()):zero()
realNameCut = {}
counterperf = 0
latitudeperformance = 0
longitudeperformance = 0
for i=1,real_data:size() do
   local groundtruthlat = real_data.labels[i][1]
   local groundtruthlon = real_data.labels[i][2]
   local predictiondata = model:forward(real_data.data[i])
   local predictionlatdata = predictiondata[1][1]
   local predictionlondata = predictiondata[1][2]
   if groundtruthlat > latmin and groundtruthlat < latmax and groundtruthlon > lonmin and groundtruthlon < lonmax then 
      class_performancelat[i]  = torch.abs((groundtruthlat - predictionlatdata)*(groundtruthlat - predictionlatdata))
      class_performancelon[i]  = torch.abs((groundtruthlon - predictionlondata)*(groundtruthlon - predictionlondata))
      latitudeperformance = latitudeperformance + class_performancelat[i]
      longitudeperformance = longitudeperformance + class_performancelon[i] 
      realNameCut[counterperf] = realName[i]
      counterperf = counterperf + 1
      intrainFOV[i] = 1;
   end
   realLat[i] = groundtruthlat
   predLat[i] = predictionlatdata
   realLon[i] = groundtruthlon
   predLon[i] = predictionlondata
end
print("REAL ERROR: ", counterperf, latitudeperformance/counterperf, longitudeperformance/counterperf)


for i=1,real_data:size() do
   print(realName[i] .. " " .. realLat[i].." "..predLat[i].." "..realLon[i].." "..predLon[i].." "..intrainFOV[i])
end

--the good stuff
--print("\n" .. sys.COLORS.red .. "The Good Stuff:")
--for i=1,real_data:size() do
--   if class_performancelat[i] < 0.01 and class_performancelon[i] < 0.01 then print(realName[i] .. " " ..realLat[i].." "..predLat[i].." "..realLon[i].." "..predLon[i]) end
--end
--
--arraytoplot = torch.Tensor(22,60)
--for i=1,22 do
--   for j=1,60 do
--      arraytoplot[i][j] = torch.sum(real_data.data[1][1][i+(j-1)*22])/150.
--   end
--end
--gnuplot.raw("unset colorbox")
--gnuplot.pngfigure('R5199E6174.png')
--gnuplot.imagesc(arraytoplot,'color')
--gnuplot.plotflush()
--gnuplot:close()
      
