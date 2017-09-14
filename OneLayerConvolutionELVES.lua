require 'nn'
require 'gnuplot'
require 'image'
require 'optim'
require 'cunn'

--THIS CODE IS JUST TO UNDERSTAND WHAT COMES OUT OF THE FIRST LAYER AFTER A CONVOLUTION AND A MAXPOOL

--load data
dataPath = "/media/kswiss/ExtraDrive1/ELVESNet/merged/"
dataAll = torch.load(dataPath..'ALLDataImage.dat')
labelAll = torch.load(dataPath..'ALLDataLocation.dat')

--insert a singleton in position 2 to add the dimension of the image channel to the dataset
modelSoliton = nn.Unsqueeze(2)
dataAll = modelSoliton:forward(dataAll)
--print(dataAll:size())

--9781 to play with --- later found two unsorted elves due to bad latitude values in label. Need to remove them from dataset. 6675 6153. done later. Also ran out of memory with a train set of 8000
trsize = 1000
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
cuda = true

--net input parameters ---> THIS NEEDS TO BE STUDIED
n_inputs_1 = 1 --number of input planes to layer 1
n_filters_1 = 16 --number of convolutional filters to layer 1
n_convx = 4 --conv filter size - use square filter for now? image not square.
n_convy = 4 --conv filter size - use square filter for now? image not square.
n_pool = 2 --size of max pooling filter
n_pool_dx = n_pool --size of steps by which to move filter.

learningRateValue = 1e-2
print_every = 10
nb_epoch = 500

latLow = -37
latHigh = -32
lonLow = -68
lonHigh = -62
nSectionsLat = 12
nSectionsLon = 12
nSectionsTot = nSectionsLat*nSectionsLon

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
print("Number of uncategorized ELVES: " .. torch.sum(torch.eq(labelAllReworked,0)))


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

--print(train_data.data:size())
print(train_data.size(), " ELVES in train dataset. Location of a train ELVES (Section): ",train_data.labels[2])
print(test_data.size(), " ELVES in test dataset.")

local model = nn.Sequential()
model:add(nn.SpatialConvolution(n_inputs_1,n_filters_1,n_convx,n_convy))
model:add(nn.Tanh())  --f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
--model:add(nn.ReLU()) --f(x) = max(0, x)
--model:add(nn.Sigmoid()) --f(x) = 1 / (1 + exp(-x))
--model:add(nn.SoftMax()) --f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)
model:add(nn.SpatialMaxPooling(n_pool,n_pool,n_pool_dx,n_pool_dx));

--===========
--CUDA
--===========
if(cuda) then
   model:cuda()
   train_data.data = train_data.data:cuda()
   train_data.labels = train_data.labels:cuda()
   test_data.data = test_data.data:cuda()
   test_data.labels = test_data.labels:cuda()
end


model:forward(train_data.data[18])
--print(train_data.data[18][1])

gnuplot.pngfigure("imageinput.png")
--gnuplot.figure(221)
gnuplot.imagesc(train_data.data[18][1])
gnuplot.raw("unset border")
gnuplot.raw("unset xtics")
gnuplot.raw("unset ytics")
gnuplot.raw("unset colorbox")
gnuplot.title("input")
gnuplot.raw('replot')
gnuplot.close()

--merge 6 filtered outputs of first conv layer:
spatToPlot1 = torch.cat(model:get(1).output[1],model:get(1).output[2],2)
spatToPlot2 = torch.cat(model:get(1).output[3],model:get(1).output[4],2)
spatToPlot3 = torch.cat(model:get(1).output[5],model:get(1).output[6],2)
spatToPlot = torch.cat(spatToPlot1,spatToPlot2,1):cat(spatToPlot3,1)
gnuplot.pngfigure("SpatConv1.png")
--gnuplot.figure(222)
gnuplot.imagesc(spatToPlot)
gnuplot.raw("unset border")
gnuplot.raw("unset xtics")
gnuplot.raw("unset ytics")
gnuplot.raw("unset colorbox")
gnuplot.title("SpatialConvolution (First)")
gnuplot.raw('replot')
gnuplot.close()


--merge 6 filtered outputs of first trans layer:
transToPlot1 = torch.cat(model:get(2).output[1],model:get(2).output[2],2)
transToPlot2 = torch.cat(model:get(2).output[3],model:get(2).output[4],2)
transToPlot3 = torch.cat(model:get(2).output[5],model:get(2).output[6],2)
transToPlot = torch.cat(transToPlot1,transToPlot2,1):cat(transToPlot3,1)
--gnuplot.figure(223)
gnuplot.pngfigure("Trans1.png")
gnuplot.imagesc(transToPlot)
gnuplot.raw("unset border")
gnuplot.raw("unset xtics")
gnuplot.raw("unset ytics")
gnuplot.raw("unset colorbox")
gnuplot.title("Transfer (First)")
gnuplot.raw('replot')
gnuplot.close()

--
--merge 6 outputs of max pool layer
poolToPlot1 = torch.cat(model:get(3).output[1],model:get(3).output[2],2)
poolToPlot2 = torch.cat(model:get(3).output[3],model:get(3).output[4],2)
poolToPlot3 = torch.cat(model:get(3).output[5],model:get(3).output[6],2)
poolToPlot = torch.cat(poolToPlot1,poolToPlot2,1):cat(poolToPlot3,1)
--gnuplot.figure(224)
gnuplot.pngfigure("SpatMaxPool1.png")
gnuplot.imagesc(poolToPlot)
gnuplot.raw("unset border")
gnuplot.raw("unset xtics")
gnuplot.raw("unset ytics")
gnuplot.raw("unset colorbox")
gnuplot.title("SpatialMaxPooling (First)")
gnuplot.raw('replot')
gnuplot.close()
