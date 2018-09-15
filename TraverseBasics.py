import os
import pydicom

# Count and identify CT and MRI images recursively in a directory

for dirpath, dirs, files in os.walk("C:/Users/hp/Downloads/tcga-lihc/TCGA-LIHC_ORIGINAL/"):	 
	path = dirpath.split('/')
	print('|', (len(path))*'---', '[',os.path.basename(dirpath),']')
	
	qt_total=0
	qt_ct=0
	qt_mr=0
	for f in files:
		qt_total = qt_total + 1

		if os.path.splitext(f)[1] == ".dcm":
			ds = pydicom.dcmread(dirpath + "/" + f)
			if ds.Modality=="CT":
				qt_ct = qt_ct + 1
			elif ds.Modality=="MR":
				qt_mr = qt_mr + 1

		print('|', len(path)*'---', f)
	
	print("Qt MR = ", qt_mr)
	print("Qt CT = ", qt_ct)
	print("Qt Total = ", qt_total)
