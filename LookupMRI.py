import os
import pydicom

# Identify MRI images recursively in a directory

for dirpath, dirs, files in os.walk("C:/Users/hp/Downloads/tcga-lihc/TCGA-LIHC_ORIGINAL/"):	 
	path = dirpath.split('/')
	
	for f in files:
		if os.path.splitext(f)[1] == ".dcm":
			ds = pydicom.dcmread(dirpath + "/" + f)
			if ds.Modality=="MR":
				print(path)
				break
