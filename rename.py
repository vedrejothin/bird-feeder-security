import shutil
import os

#dirname = "images/squirrel/"
dirname = "images/test/"
files = os.listdir(dirname)

curr = 0

files.sort()
for f in files:
	l = f.replace("frame", "")
	l = l.replace(".jpg", "")
	curr_string = str(curr)
	
	curr_string = "frame"+"".join(["0" for k in range(4-len(curr_string))])+curr_string+".jpg"
	#print f, "\t",curr_string
	shutil.move(dirname+f, dirname+curr_string)
	
	curr = curr+1
	
