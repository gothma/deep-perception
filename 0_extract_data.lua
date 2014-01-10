require 'image'
require 'torch'
require 'lib/kitti_tools'

trainData = {
	   data = torch.DoubleTensor(80256,3,32,32),
	   labels = torch.LongStorage(80265)
	   }


for i =0, 7481 do 

img = read_image('data/images/training/image_2' , i .. '.png')
labels = read_labels('data/labels/labels_2',i)




end


