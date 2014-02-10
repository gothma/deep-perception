set xlabel 'Original Label'
set ylabel 'Predicted Label'
set size square
set key off
 
set terminal png size 1000,600 enhanced font "Helvetica,20"
set output fname.'.png'

#XTICS="`awk 'BEGIN{getline}{printf "%s ",$1}' eheader.txt`"
YTICS="`head -1 labels.txt`"

set for [i=1:words(YTICS)] xtics ( word(YTICS,i) i-1 )

set for [i=1:words(YTICS)] ytics ( word(YTICS,i) i-1 )

#plot "<awk '{$1=\"\"}1' fname.dat | sed '1 d'" matrix w image


#plot "fname | sed '1 d'" matrix w image
set palette model RGB maxcolors 200
set cbrange [0:200]
set cblabel "Number of prediction"

set xtic font ",12"
set ytic font ",12"
set xtics rotate by -90	
#set grid xtics ytics ls 100

print(fname)
plot fname matrix with image  

pause -1

