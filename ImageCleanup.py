import os
import pydicom


# Remove all dicom images recursively, keeping only axial studies.

SRC_DIR="C:/Users/hp/Downloads/tcga-lihc/TCGA-LIHC_CT_DCM/"

for dirpath, dirs, files in os.walk(SRC_DIR):	 
	path = dirpath.split('/')
	
	for f in files:
		if os.path.splitext(f)[1] == ".dcm":
			ds = pydicom.dcmread(dirpath + "/" + f)
			
			if any("axial" in s.lower() for s in ds.ImageType):
				print(ds.ImageType)
			else:
				print(dirpath + "/" + f)
				print("Delete!")
				os.remove(dirpath + "/" + f)
	
	if len(os.listdir(dirpath + "/")) == 0:
            os.rmdir(dirpath + "/")
