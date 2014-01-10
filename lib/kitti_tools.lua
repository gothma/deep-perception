 
require 'image'
require 'io'

--- read_image: 
--- read image with filename composed out of path and number and return it
read_image = function (path, number)
  local filename = string.format ('%s/%06d.png',path,number) 
  return image.load(filename), filename
end


--- read_labels: 
--- read label file with filename composed out of path and number and 
--- return objects
read_labels = function (path, number)
  local filename = string.format ('%s/%06d.txt',path,number)
  local objects = {}
  for line in io.lines(filename) do
    local n = #objects+1
    objects[n] = {}
    local tokens = {}
    for token in string.gmatch(line, "[^%s]+") do
      tokens[#tokens+1] = token
    end
    objects[n].type = tokens[1]
    objects[n].truncated = tokens[2]
    objects[n].occluded = tokens[3]
    objects[n].alpha = tokens[4]
    objects[n].x1 = tokens[5]+1
    objects[n].y1 = tokens[6]+1
    objects[n].x2 = tokens[7]+1
    objects[n].y2 = tokens[8]+1
    objects[n].height = tokens[9]
    objects[n].width = tokens[10]
    objects[n].length = tokens[11]
    objects[n].x = tokens[12]
    objects[n].y = tokens[13]
    objects[n].z = tokens[14]
    objects[n].rotation_y = tokens[15]
    objects[n].score = tokens[16]
  end
  return objects, filename
end


--- write_labels: 
--- write object list to label file with filename composed out of path and 
--- number and return objects
write_labels = function (path, number, objects)
  local filename = string.format ('%s/%06d.txt',path,number)
  file = io.open(filename, "w")
  for k,object in pairs(objects) do
    local line = object.type .. ' ' .. tostring(object.truncated or -1) .. ' ' .. tostring(object.occluded or -1) .. ' ' .. tostring(object.alpha or -1) .. ' ' .. tostring(object.x1-1 or -1) .. ' ' .. tostring(object.y1-1 or -1) .. ' ' .. tostring(object.x2-1 or -1) .. ' ' .. tostring(object.y2-1 or -1) .. ' ' .. tostring(object.height or -1) .. ' ' .. tostring(object.width or -1) .. ' ' .. tostring(object.length or -1) .. ' ' .. tostring(object.x or -1) .. ' ' .. tostring(object.y or -1) .. ' ' .. tostring(object.z or -1) .. ' ' .. tostring(object.rotation_y or -1) .. ' ' .. tostring(object.score or 0) .. '\n'
    file:write (line )
  end
  file:close ()
end


--- add_labels_to_image: 
--- draw colored rectangles into a copy of the image, one for each object
add_labels_to_image = function (srcimage, objects, colormap)
  local nchannels = srcimage:size()[1]
  local nrows = srcimage:size()[2]
  local ncols = srcimage:size()[3]
  local destimage
  if (nchannels==1) then
    destimage = torch.Tensor(3, nrows, ncols)
    destimage:select (1,1):copy(srcimage)
    destimage:select (1,2):copy(srcimage)
    destimage:select (1,3):copy(srcimage)
  else
    destimage = torch.Tensor(3, nrows, ncols):copy (srcimage)
  end
  for k,v in pairs(objects) do
    color = colormap[v.type]
    if (not color) then
      color = { ['r']=0, ['g']=0, ['b']=0 };
    end
    if (v.x1<=1) then v.x1=1 end
    if (v.x2<=2) then v.x2=2 end
    if (v.y1<=1) then v.y1=1 end
    if (v.y2<=2) then v.y2=2 end
    if (v.x1>=ncols-1) then v.x1=ncols-1 end
    if (v.x2>=ncols) then v.x2=ncols end
    if (v.y1>=nrows-1) then v.y1=nrows-1 end
    if (v.y2>=nrows) then v.y2=nrows end
    for x = v.x1, v.x2 do
      destimage[1][v.y1][x] = color.r
      destimage[2][v.y1][x] = color.g
      destimage[3][v.y1][x] = color.b
      destimage[1][v.y2][x] = color.r
      destimage[2][v.y2][x] = color.g
      destimage[3][v.y2][x] = color.b
      destimage[1][v.y1+1][x] = color.r
      destimage[2][v.y1+1][x] = color.g
      destimage[3][v.y1+1][x] = color.b
      destimage[1][v.y2-1][x] = color.r
      destimage[2][v.y2-1][x] = color.g
      destimage[3][v.y2-1][x] = color.b
    end
    for y = v.y1, v.y2 do
      destimage[1][y][v.x1] = color.r
      destimage[2][y][v.x1] = color.g
      destimage[3][y][v.x1] = color.b
      destimage[1][y][v.x2] = color.r
      destimage[2][y][v.x2] = color.g
      destimage[3][y][v.x2] = color.b
      destimage[1][y][v.x1+1] = color.r
      destimage[2][y][v.x1+1] = color.g
      destimage[3][y][v.x1+1] = color.b
      destimage[1][y][v.x2-1] = color.r
      destimage[2][y][v.x2-1] = color.g
      destimage[3][y][v.x2-1] = color.b
    end
  end
  return destimage
end

colormap_kitti = {}
colormap_kitti.Car = {}
colormap_kitti.Car.r = 1
colormap_kitti.Car.g = 0
colormap_kitti.Car.b = 0
colormap_kitti.Van = {}
colormap_kitti.Van.r = 0.7
colormap_kitti.Van.g = 0
colormap_kitti.Van.b = 0.4
colormap_kitti.Truck = {}
colormap_kitti.Truck.r = 0.4
colormap_kitti.Truck.g = 0
colormap_kitti.Truck.b = 0.7
colormap_kitti.Pedestrian = {}
colormap_kitti.Pedestrian.r = 0
colormap_kitti.Pedestrian.g = 0.8
colormap_kitti.Pedestrian.b = 0
colormap_kitti.Tram = {}
colormap_kitti.Tram.r = 0.5
colormap_kitti.Tram.g = 0.5
colormap_kitti.Tram.b = 0
colormap_kitti.Person_sitting = {}
colormap_kitti.Person_sitting.r = 0.2
colormap_kitti.Person_sitting.g = 0.7
colormap_kitti.Person_sitting.b = 1
colormap_kitti.Cyclist = {}
colormap_kitti.Cyclist.r = 0.1
colormap_kitti.Cyclist.g = 0.1
colormap_kitti.Cyclist.b = 1
colormap_kitti.Misc = {}
colormap_kitti.Misc.r = 1
colormap_kitti.Misc.g = 0.7
colormap_kitti.Misc.b = 0.2
colormap_kitti.DontCare = {}
colormap_kitti.DontCare.r = 0.7
colormap_kitti.DontCare.g = 0.7
colormap_kitti.DontCare.b = 0.7


--- Usage:
--- for KITTI data:
--[[
i, n = read_image ('/home/lauer/Data/KITTI/2012_object/training/image_2', 8)
labels = read_labels ('/home/lauer/Data/KITTI/2012_object/training/label_2', 8)
write_labels ('/tmp', 8, labels)
j = add_labels_to_image (i, labels, colormap_kitti)
image.display(j)
---]]
