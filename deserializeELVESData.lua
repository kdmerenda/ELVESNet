datapath = "/media/kswiss/ExtraDrive1/ELVESNet/mergeddata/data/"
outpath = "/media/kswiss/ExtraDrive1/ELVESNet/mergeddata/out/"
counter = 0
for filename in paths.iterfiles(datapath) do
   counter = counter+1
   file = torch.DiskFile(datapath..filename, 'r')
   npixels = file:readInt()
   ntimebins = file:readInt()
   file:close()
   namelength = string.len(filename)
end
namelength = torch.Tensor(counter)
counter = 0
for filename in paths.iterfiles(datapath) do
   counter = counter + 1
   namelength[counter] = string.len(filename)
end


print(counter .. " files found with size ".. npixels .. " x " .. ntimebins)

dataALL = torch.Tensor(counter,npixels,ntimebins)
traceIntegralALL =  torch.Tensor(counter,npixels) 
latlon = torch.Tensor(counter,2)
imageIntegral = torch.Tensor(counter,22,60)
filenameAll = torch.IntTensor(counter,torch.max(namelength))

print(dataALL:size(), latlon:size())
counter = 1
require("gnuplot")
for filename in paths.iterfiles(datapath) do
   print(filename)
   for ichar=1,filenameAll:size(2) do
      filenameAll[counter][ichar] = 32
      if string.byte(filename,ichar)~=nil then filenameAll[counter][ichar] = string.byte(filename,ichar) end
   end
   --   print(filenameAll[counter])
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

   local handle = io.popen("grep -i \"".. filename .. "\" /media/kswiss/ExtraDrive1/ELVESNet/mergeddata/TruthMerged.list")
   local result = handle:read("*a")
   handle:close()
   result = string.gsub(result,"/media/kswiss/ExtraDrive1/ELVESNet/mergeddata/"..filename.." ","")
   subcounter = 1
   print(result)
   for strNumber in string.gmatch(result, '%-?%d+%.%d+') do
      latlon[counter][subcounter] = tonumber(strNumber)
      subcounter = subcounter + 1
   end
   print(latlon[counter])
   counter = counter+1
--   if counter==15 then break end
end
--dump all the whole tensor in one struct to gt easy access in future code.
dataALL = dataALL:float()
imageIntegral = imageIntegral:float()
latlon = latlon:float()
print(dataALL:size(),dataALL:type())
torch.save(outpath..'ALLDataTensor.dat', dataALL)
torch.save(outpath..'ALLDataImage.dat', imageIntegral)
torch.save(outpath..'ALLDataLocation.dat', latlon)
torch.save(outpath..'ALLDataName.dat', filenameAll)
