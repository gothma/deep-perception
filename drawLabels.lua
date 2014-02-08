require 'lib/kitti_tools'
require 'image'

cmd = torch.CmdLine()

cmd:option('-image', '1', 'Image to draw the labels in')

opt = cmd:parse(arg or {})

i = read_image('data/images/testing/image_2', opt.image)
l = read_labels('results/test', opt.image)
j = add_labels_to_image(i, l, colormap_kitti)
image.display(j)

