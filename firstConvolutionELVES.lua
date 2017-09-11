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
      for j=1,nSectionsLon do
	 latLowTmp  = latHigh - (i)*(torch.abs(latLow-latHigh) / nSectionsLat) 
	 latHighTmp = latHigh - (i-1)*(torch.abs(latLow-latHigh) / nSectionsLat)
	 lonLowTmp  = lonLow + (j-1)*(torch.abs(lonLow-lonHigh) / nSectionsLon) 
	 lonHighTmp = lonLow + (j)*(torch.abs(lonLow-lonHigh) / nSectionsLon)
	 if labelAll[k][2] >= lonLowTmp and labelAll[k][2] <= lonHighTmp and labelAll[k][1] >= latLowTmp and labelAll[k][1] <= latHighTmp then
	    labelAllReworked[k] = j+(i-1)*nSectionsLon
	 end
      end
   end
   if labelAllReworked[k] == 0 then  print(labelAll[k],k) end 
end

gnuplot.raw("set xrange [0.5:4.5]")
gnuplot.hist(labelAllReworked,40)   
print("Number of uncategorized ELVES: " .. torch.sum(torch.eq(labelAllReworked,0)) .. "\n\n\n")

--save the data in a struct and format the train and test sets to get part of the full dataset
train_data = {
   data = dataAll[{ {1,trsize},{},{} }],
   labels = labelAll[{ {1,trsize} }],
   size = function() return trsize end
}
test_data = {
   data = dataAll[{ {trsize+1,trsize+tesize},{},{} }],
   labels = labelAll[{ {trsize+1,trsize+tesize} }],
   size = function() return tesize end
}

print(train_data.size(), " Location of train ELVES (lat,lon): ",train_data.labels[2][1], train_data.labels[2][2])
print(test_data.size(), " Location of test ELVES (lat,lon): ",test_data.labels[2][1], test_data.labels[2][2])

-- all images here are the integrated signal in time. So take one of the 9781 images and test a possible comvolution on a 'grayscale'
imageSample = torch.Tensor(1,dataAll:size()[2],dataAll:size()[3])
imageSample[1] = dataAll[13]
--print(imageSample)
--image.display(imageSample)--run with qlua for this...
gnuplot.raw("set xrange restore")
gnuplot.raw("unset border")
gnuplot.raw("unset xtics")
gnuplot.raw("unset ytics")
gnuplot.raw("unset colorbox")
--gnuplot.imagesc(imageSample[1])

net = nn.SpatialConvolution(1,18,5,5)
--gnuplot.imagesc(net.weight)
--image.display(net.weight)

res = net:forward(imageSample)
print("\n\n")
print(res:size())
--image.display(res)


