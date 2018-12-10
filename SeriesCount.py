import os

# Count CT series and files recursively

#SRC_DIR="C:/Users/hp/Downloads/tcga-lihc/TCGA-LIHC_CT_DCM/"
#SRC_DIR="C:/Users/hp\Downloads/tcga-kirp/TCGA-KIRP_CT/"
#SRC_DIR="C:/Users/hp/Downloads/tcga-stad/TCGA-STAD/"
SRC_DIR="C:/Users/hp/Downloads/cptac-pda/CPTAC-PDA/"

qtd = 0
pathlist=[]

for dirpath, dirs, files in os.walk(SRC_DIR):	 
	path = dirpath.split('/')
	
	for f in files:
		if os.path.splitext(f)[1] == ".dcm":
                    qtd = qtd + 1

                    if not(any(dirpath in s for s in pathlist)):
                        pathlist.append(dirpath)


print(pathlist[0])
print(pathlist[1])
print(pathlist[5])
print("Files = ", qtd)
print("Series = ", len(pathlist))
                   

