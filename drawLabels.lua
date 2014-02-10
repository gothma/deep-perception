require 'lib/kitti_tools'
require 'image'


if #arg ~= 2 then
  print('Usage: drawLabels')
end

i = read_image('data/images/testing/image_2', arg[1])
l = read_labels('results/test', arg[1])
j = add_labels_to_image(i, l, colormap_kitti)
image.display(j)

