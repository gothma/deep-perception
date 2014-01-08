require 'image'
require 'nn'
loaded= image.load('6.png')

 local w_patch = 20
 local h_patch = 20
 img = loaded[1]:double()
 trainData = { data = torch.DoubleTensor(10000,1,20,20),
               lables = torch.ByteTensor(10000):fill(5)} 
 for t = 1,10000 do 
 x= torch.random(img:size(1)-20)
 y= torch.random(img:size(2)-20)
   trainData.data[t][1] = (img[{{x , x + w_patch -1},{y, y + h_patch - 1}}]*255):byte():double()
 end
 torch.save('smiley_non6.t7',trainData)

loaded= image.load('1.png')

 local w_patch = 20
 local h_patch = 20
 img = loaded[1]:double()
 trainData = { data = torch.DoubleTensor(10000,1,20,20),
               lables = torch.ByteTensor(10000):fill(5)} 
 for t = 1,10000 do 
 x= torch.random(img:size(1)-20)
 y= torch.random(img:size(2)-20)
   trainData.data[t][1] = (img[{{x , x + w_patch -1},{y, y + h_patch - 1}}]*255):byte():double()

 end
 torch.save('smiley_non1.t7',trainData)

loaded= image.load('4.png')

 local w_patch = 20
 local h_patch = 20
 img = loaded[1]:double()
 trainData = { data = torch.DoubleTensor(10000,1,20,20),
               lables = torch.ByteTensor(10000):fill(5)} 
 for t = 1,10000 do 
 x= torch.random(img:size(1)-20)
 y= torch.random(img:size(2)-20)
   trainData.data[t][1] = (img[{{x , x + w_patch -1},{y, y + h_patch - 1}}]*255):byte():double()
 end
 torch.save('smiley_no4.t7',trainData)

loaded= image.load('3.png')

 local w_patch = 20
 local h_patch = 20
 img = loaded[1]:double()
 trainData = { data = torch.DoubleTensor(10000,1,20,20),
               lables = torch.ByteTensor(10000):fill(5)} 
 for t = 1,10000 do 
 x= torch.random(img:size(1)-20)
 y= torch.random(img:size(2)-20)
   trainData.data[t][1] = (img[{{x , x + w_patch -1},{y, y + h_patch - 1}}]*255):byte():double()
 end
 torch.save('smiley_no3.t7',trainData)


loaded= image.load('2.png')

 local w_patch = 20
 local h_patch = 20
 img = loaded[1]:double()
 trainData = { data = torch.DoubleTensor(10000,1,20,20),
               lables = torch.ByteTensor(10000):fill(5)} 
 for t = 1,10000 do 
 x= torch.random(img:size(1)-20)
 y= torch.random(img:size(2)-20)
   trainData.data[t][1] = (img[{{x , x + w_patch -1},{y, y + h_patch - 1}}]*255):byte():double()
 end
 torch.save('smiley_non2.t7',trainData)
