 
-- return the maximum of x and y
max = function (x, y)
  if (x<y) then 
    return y
  else 
    return x
  end
end

-- return the minimum of x and y
min = function (x, y)
  if (x>y) then 
    return y
  else 
    return x
  end
end

-- return the amount of overlap of obj1 and obj2
-- 0: no overlap
-- 1: both areas are identical
-- overlap_ratio = size(intersection)/size(union)
-- obj must have the attributes x1, x2, y1, y2
overlap_area = function (obj1, obj2)
  local x1 = max(obj1.x1, obj2.x1);
  local x2 = min(obj1.x2, obj2.x2);
  local y1 = max(obj1.y1, obj2.y1);
  local y2 = min(obj1.y2, obj2.y2);
  if (x1<x2 and y1<y2) then
    local size1 = (obj1.x2-obj1.x1)*(obj1.y2-obj1.y1)
    local size2 = (obj2.x2-obj2.x1)*(obj2.y2-obj2.y1)
    return ((x2-x1+1)*(y2-y1+1))/((obj1.x2-obj1.x1+1)*(obj1.y2-obj1.y1+1)+(obj2.x2-obj2.x1+1)*(obj2.y2-obj2.y1+1)-(x2-x1+1)*(y2-y1+1))
  else
    return 0
  end
end

-- calculate the local maxima in score of objects and
-- return the list of local maxima
-- objects is a list of objects. Each object has at
-- least the following attributes:
-- x1, x2, y1, y2: the bounding box of the object
-- score: a confidence measure of the object classification
-- type: the class label of the object
--
-- Suppression strategy:
-- if two object of the same type overlap more than 40% the one with the 
-- lower score is suppressed
nonmaxima_suppression = function (objects)
  for i=1,#objects do
    objects[i].is_maximum = true
  end
  for i=1,#objects do
    for j=i+1,#objects do
      local overlap = overlap_area (objects[i], objects[j])
      if (overlap>0.2) then -- and objects[i].type==objects[j].type) then
        if (objects[i].score>objects[j].score) then
          objects[j].is_maximum = false
        else
          objects[i].is_maximum = false
        end
      end
    end
  end
  local new_objects = {}
  local i = 1
  for j=1,#objects do
    if (objects[j].is_maximum) then
      new_objects[i]=objects[j]
      i = i+1
    end
  end
  return new_objects
end

-------------------------------------------------------------------
----- merge the 

merge_objects = function (obj1 , obj2)

  local objOut = {} 
  local area1 = (obj1.x2 - obj1.x1) * (obj1.y2 - obj1.y1)
  local area2 = (obj2.x2 - obj2.x1) * (obj2.y2 - obj2.y1)
  
  if (area1 > area2) then 
     objOut = obj1
    --[=[ if (objOut.x1 > obj2.x1) then
       objOut.x1 = obj2.x1
     end 
     if (objOut.x2 < obj2.x2) then
       objOut.x2 = obj2.x2
     end
     if (objOut.y1 > obj2.y1) then
       objOut.y1 = obj2.y1
     end 
     if (objOut.y2 < obj2.y2) then
       objOut.y2 = obj2.y2
     end--]=]
  else 
     objOut = obj2
    --[=[ if (objOut.x1 > obj1.x1) then
       objOut.x1 = obj1.x1
     end 
     if (objOut.x2 < obj1.x2) then
       objOut.x2 = obj1.x2
     end
     if (objOut.y1 > obj1.y1) then
       objOut.y1 = obj1.y1
     end 
     if (objOut.y2 < obj1.y2) then
       objOut.y2 = obj1.y2
     end ]=]
  end 
  return objOut
end

nonmaxima_merge = function (objects)
  for i=1,#objects do
    objects[i].is_maximum = true
  end
  for i=1,#objects do
    for j=1,#objects do
      if (objects[j].is_maximun) then
        local overlap = overlap_area (objects[i], objects[j])
        if (overlap>0.2) then -- and objects[i].type==objects[j].type) then
          objects[j].is_maximum = false
          objects[i].is_maximum = merge_objects(objects[i], objects[j])
        end
      end
    end
  end
  local new_objects = {}
  local i = 1
  for j=1,#objects do
    if (objects[j].is_maximum) then
      new_objects[i]=objects[j]
      i = i+1
    end
  end
  return new_objects
end
