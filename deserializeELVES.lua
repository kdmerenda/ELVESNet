datapath = "/media/kswiss/ExtraDrive1/ELVESNet/merged/data/"
outpath = "/media/kswiss/ExtraDrive1/ELVESNet/merged/out/"
counter = 0
for filename in paths.iterfiles(datapath) do
   counter = counter+1
   file = torch.DiskFile(datapath..filename, 'r')
   npixels = file:readInt()
   ntimebins = file:readInt()
   file:close()
end
print(counter .. " files found with size ".. npixels .. " x " .. ntimebins)

dataALL = torch.Tensor(counter,npixels,ntimebins)
traceIntegralALL =  torch.Tensor(counter,npixels) 
latlon = torch.Tensor(counter,2)
imageIntegral = torch.Tensor(counter,22,60)
print(dataALL:size(), latlon:size())
counter = 1
require("gnuplot")
for filename in paths.iterfiles(datapath) do
   print(filename)
   file2 = torch.DiskFile(datapath..filename, 'r')
   npixels = file2:readInt()
   ntimebins = file2:readInt()
   for i=1,npixels do
      dataALL[counter][i] = torch.Tensor(file2:readDouble(ntimebins))
      traceIntegralALL[counter][i] = torch.sum(dataALL[counter][i])/ntimebins
   end
   arraytoplot = torch.Tensor(22,60)
   for i=1,22 do
      for j=1,60 do
	 arraytoplot[i][j] = traceIntegralALL[counter][i+(j-1)*22]
      end
   end
   imageIntegral[counter] = arraytoplot

   if counter<=15 then
      gnuplot.raw("unset colorbox")
      gnuplot.pngfigure(outpath..filename..'.png')
      gnuplot.imagesc(arraytoplot,'color')
      gnuplot.plotflush()
      gnuplot:close()
   end

   file2:close()
   local handle = io.popen("grep -i \"".. filename .. "\" /media/kswiss/ExtraDrive1/ELVESNet/merged/TruthMerged.list")
   local result = handle:read("*a")
   handle:close()
   result = string.gsub(result,"/media/kswiss/ExtraDrive1/ELVESNet/merged/"..filename.." ","")
   subcounter = 1;
   for strNumber in string.gmatch(result, '%-?%d+%.%d+') do
      latlon[counter][subcounter] = tonumber(strNumber)
      subcounter = subcounter + 1
   end
--   print(filename, latlon[counter])
   counter = counter+1
   if counter==15 then break end
end

--dump all the whole tensor in one struct to gt easy access in future code. 
--torch.save(outpath..'ALLDataTensor.dat', dataALL)
--torch.save(outpath..'ALLDataLocation.dat', latlon)
torch.save(outpath..'ALLDataImage.dat', imageIntegral)
