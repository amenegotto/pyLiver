import os
import pydicom


# Search for CT images in SRC_DIR
# and create DEST_DIR studies 
# This extract the images without filtering, just for a first analysis (before pre-processing phase)

SRC_DIR="C:/Users/hp/Downloads/tcga-lihc/TCGA-LIHC_ORIGINAL"
DEST_DIR="C:/Users/hp/Downloads/tcga-lihc/TCGA-LIHC_CT"

if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

qtlist=[]
pathlist=[]

for dirpath, dirs, files in os.walk(SRC_DIR):	 
	path = dirpath.split('/')
	
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
	
	if qt_ct > 0:
		qtlist.append(qt_ct)
		pathlist.append(dirpath)

for path in pathlist:
	s = path.split("/")
	study=s[5]
	basedir=DEST_DIR + '/' + study
	nextid=1
	if not os.path.exists(basedir):
		os.makedirs(basedir)

	while (1>0):
		currid=basedir + '/' + str(nextid)
		if not os.path.exists(currid):
			os.makedirs(currid)
			break
		nextid = nextid + 1

	for f in os.listdir(path):
		os.system("cp " + path.replace(" ", "\ ") + "/" + f + " " + currid + "/" + f)
