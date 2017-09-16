require 'nn'
require 'gnuplot'
require 'image'
require 'optim'
require 'cunn'


--load data
dataPath = "/media/kswiss/ExtraDrive1/ELVESNet/merged/"
dataAll = torch.load(dataPath..'ALLDataImage.dat')
labelAll = torch.load(dataPath..'ALLDataLocation.dat')

--insert a singleton in position 2 to add the dimension of the image channel to the dataset
modelSoliton = nn.Unsqueeze(2)
dataAll = modelSoliton:forward(dataAll)
--print(dataAll:size())

-- this seems intricate to deal with... for some reason torch uses doubles to initialize all model..
--torch.setdefaulttensortype('torch.FloatTensor')

--9781 to play with --- later found two unsorted elves due to bad latitude values in label. Need to remove them from dataset. 6675 6153. done later. Also ran out of memory with a train set of 8000]
--numLoops to increment train size and numSubLoop to repeat process x times at each train size. 
numLoops = 14
numSubLoops = 5
trsize = torch.Tensor(numLoops)
for i=1,numLoops do
   trsize[i] = 500 + (i-1)*500
end
trerrolat = torch.Tensor(numLoops,numSubLoops):zero()
teerrolat = torch.Tensor(numLoops,numSubLoops):zero()
trerrolon = torch.Tensor(numLoops,numSubLoops):zero()
teerrolon = torch.Tensor(numLoops,numSubLoops):zero()
tesize = 1000

--fixing the problem
data1 = dataAll[{ {1,6153-1},{},{} }]
label1 = labelAll[{ {1,6153-1},{} }]
data2 = dataAll[{ {6153+1,6675-1},{},{} }]
label2 = labelAll[{ {6153+1,6675-1},{} }]
data3 = dataAll[{ {6675+1,dataAll:size()[1]},{},{} }]
label3 = labelAll[{ {6675+1,dataAll:size()[1]},{} }]

dataAll = torch.cat(data1,data2,1):cat(data3,1)
labelAll = torch.cat(label1,label2,1):cat(label3,1)
--9779 ELVES left to play with.

--gpu work
cuda = false

--plots
plotsPerEpoch = false
plotsPerLayer = false

nb_epoch = 100
learningRateValue = 1e-2

--====
--LOOP
--====
for jj=1,numLoops do
   for kk=1,numSubLoops do
      print(trsize[jj], kk)
      
      train_data = {}
      test_data = {}
      
      counter = 0
      for i=1,trsize[jj] do
	 local input=torch.Tensor(dataAll[i])
	 --take advantage of loop over data to remove blank images
	 if torch.mean(input)==0 then
	    counter = counter + 1
	 end
	 local output=torch.Tensor(labelAll[i])
	 if torch.mean(input)~=0 then train_data[i-counter] = {input,output} end
--	 print(train_data[i-counter])
      end
      print(counter.." empty images in train set")
      function train_data:size() return trsize[jj]-counter end

      counterte = 0
      for i=trsize[jj]+1,trsize[jj]+1+tesize do
	 local input=torch.Tensor(dataAll[i])
	 if torch.mean(input)==0 then
	    counterte = counterte + 1
	 end
	 local output=torch.Tensor(labelAll[i])
	 if torch.mean(input)~=0 then test_data[i-counterte-(trsize[jj]+1)] = {input,output} end
      end      
      print(counterte.." empty images in test set")
      function test_data:size() return tesize-counterte end

      --final sizes
      print(train_data:size(), " ELVES in train dataset.")
      print(test_data:size(), " ELVES in test dataset.")

      
      --save the data in a struct and format to test the nn after... diff than what Stochastic gradient requires
      traindata = {
	 data = dataAll[{ {1,trsize[jj]},{},{},{} }],
	 labels = labelAll[{ {1,trsize[jj]} }],
	 size = function() return trsize[jj] end
      }
      testdata = {
	 data = dataAll[{ {trsize[jj]+1,trsize[jj]+tesize},{},{},{} }],
	 labels = labelAll[{ {trsize[jj]+1,trsize[jj]+tesize} }],
	 size = function() return tesize end
      }

      ---------------------------
      --Container:
      ---------------------------
      --to be able to plug layers in a feed-forward fully connected manner
      local model = nn.Sequential()
      
      --===========
      --ONLY LAYER
      --===========
      --mlp
      
      model:add(nn.Reshape(22*60))
      model:add(nn.Linear(22*60, 128))
--      model:add(nn.Tanh())
      model:add(nn.Sigmoid())
--      model:add(nn.ReLU()) 
      --model:add(nn.Dropout(0.5));
      model:add(nn.Linear(128, 2))

      --==============
      -- LOSS FUNCTION
      --==============
      local criterion = nn.MSECriterion()
      print(model)
      
      --======
      -- TRAIN
      --======
      trainer = nn.StochasticGradient(model, criterion)
      trainer.maxIteration = nb_epoch
      trainer.learningRate = learningRateValue
      trainer.verbose=false
      trainer:train(train_data)

      --classification error on train set -- not sure how to do that. will just do ((E-O)/E)^2
      class_performancelat = torch.Tensor(traindata:size()-counter):zero()
      class_performancelon = torch.Tensor(traindata:size()-counter):zero()
      counterperformance = 0
      for i=1,traindata.size() do
      	 local groundtruthlat = traindata.labels[i][1]
      	 local groundtruthlon = traindata.labels[i][2]
	 if torch.mean(traindata.data[i]) ~= 0 then
	    local prediction = model:forward(traindata.data[i])
	    local predictionlat = prediction[1][1]
	    local predictionlon = prediction[1][2]
	    class_performancelat[i-counterperformance]  = torch.abs((groundtruthlat - predictionlat)/groundtruthlat)
	    class_performancelon[i-counterperformance]  = torch.abs((groundtruthlon - predictionlon)/groundtruthlon)
	    counterperformance = counterperformance+1
	 end
      end
      print("TRAIN ERROR: ",counterperformance, torch.mean(class_performancelat), torch.mean(class_performancelon))
      trerrolat[jj][kk] = torch.mean(class_performancelat);
      trerrolon[jj][kk] = torch.mean(class_performancelon);

      --======
      -- TEST
      --======
      class_performancelat = torch.Tensor(testdata:size()-counter):zero()
      class_performancelon = torch.Tensor(testdata:size()-counter):zero()
      counterperformance = 0
      for i=1,testdata.size() do
      	 local groundtruthlat = testdata.labels[i][1]
      	 local groundtruthlon = testdata.labels[i][2]
	 if torch.mean(testdata.data[i]) ~= 0 then
	    local prediction = model:forward(testdata.data[i])
	    local predictionlat = prediction[1][1]
	    local predictionlon = prediction[1][2]
	    class_performancelat[i-counterperformance]  = torch.abs((groundtruthlat - predictionlat)/groundtruthlat)
	    class_performancelon[i-counterperformance]  = torch.abs((groundtruthlon - predictionlon)/groundtruthlon)
--	    print(groundtruthlat,predictionlat,  class_performancelat[i-counterperformance] )
	    counterperformance = counterperformance+1
	 end
      end
      print("TEST ERROR: ", counterperformance, torch.mean(class_performancelat), torch.mean(class_performancelon))
      teerrolat[jj][kk] = torch.mean(class_performancelat);
      teerrolon[jj][kk] = torch.mean(class_performancelon);
      
      model = nil
      traindata = nil
      testdata = nil
      train_data = nil
      test_data = nil
      criterion = nil

      collectgarbage()
   end
end   
print(torch.mean(trerrolat,2),torch.mean(teerrolat,2))
print(torch.mean(trerrolon,2),torch.mean(teerrolon,2))
print(torch.std(trerrolat,2),torch.std(teerrolat,2))
print(torch.std(trerrolon,2),torch.std(teerrolon,2))
gnuplot.pngfigure("learningcurvelinear.png")
gnuplot.plot({'train lat',torch.mean(trerrolat,2)},{'train lon',torch.mean(trerrolon,2)},{'test lat',torch.mean(teerrolat,2)},{'test lon',torch.mean(teerrolon,2)})
gnuplot.raw("set xlabel 'Training Set Size (x500)'")
gnuplot.raw("set ylabel 'Mean Error'")
gnuplot.title("Learning Curve Mean (5 repeats)")
gnuplot.raw("replot")
gnuplot.close()
gnuplot.pngfigure("learningcurvelinearerror.png")
gnuplot.plot({'train lat',torch.std(trerrolat,2)},{'train lon',torch.std(trerrolon,2)},{'test lat',torch.std(teerrolat,2)},{'test lon',torch.std(teerrolon,2)})
gnuplot.raw("set xlabel 'Training Set Size (x500)'")
gnuplot.raw("set ylabel 'Error Std.'")
gnuplot.title("Learning Curve Std (5 repeats)")
gnuplot.raw("replot")
gnuplot.close()
   

