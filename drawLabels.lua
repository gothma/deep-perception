require 'lib/kitti_tools'
require 'image'
require 'xlua'

for i, ind in ipairs(arg) do
    xlua.progress(i, #arg)
    img = read_image('data/images/training/image_2', ind)
    labels = read_labels('data/training/label_2', ind)
    annotated = add_labels_to_image(img, labels, colormap_kitti)
    image.savePNG('results/' .. ind .. '.png', annotated)
end

