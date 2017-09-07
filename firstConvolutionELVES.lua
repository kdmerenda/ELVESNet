require 'nn'
require 'gnuplot'
require 'image'

--load data
dataPath = "/media/kswiss/ExtraDrive1/ELVESNet/merged/"
dataAll = torch.load(dataPath..'ALLDataImage.dat')
labelAll = torch.load(dataPath..'ALLDataLocation.dat')

--9781 to play with
trsize = 1000
tesize = 7000

--save the data in a struct and format the train and test sets
train_data = {
   data = dataAll,
   labels = labelAll,
   size = function() return dataAll:size()[1] end
}

print(train_data.size(), " Location of sample ELVES (lat,lon): ",train_data.labels[2][1], train_data.labels[2][2])

-- all images here are the integrated signal in time. So take one of the 9781 images and test a possible comvolution on a 'grayscale'
imageSample = torch.Tensor(1,dataAll:size()[2],dataAll:size()[3])
imageSample[1] = dataAll[13]
--print(imageSample)
image.display(imageSample)--run with qlua for this...
gnuplot.raw("unset border")
gnuplot.raw("unset xtics")
gnuplot.raw("unset ytics")
gnuplot.raw("unset colorbox")
--gnuplot.imagesc(imageSample[1])

net = nn.SpatialConvolution(1,18,5,5)
--gnuplot.imagesc(net.weight)
--image.display(net.weight)

res = net:forward(imageSample)
print(res:size())
image.display(res)
