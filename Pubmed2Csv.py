from Bio import Entrez, Medline

Entrez.email = "alan.menegotto@gmail.com"
count = 1
dst = open("/tmp/pmc.csv", "w")

#for i in range(0,4):
search_handle = Entrez.esearch(db="pmc", usehistory="y", term='Multimodal AND "Deep Learning" AND (cancer OR tumour OR neoplasm)', retmax=400, retstart=0)
page_record = Entrez.read(search_handle)

for pmcid in page_record['IdList']:
    print("Fetching pmcid = " + pmcid)
    fetch_handle = Entrez.efetch(db='pmc', rettype="medline", retmode="text", id=pmcid)
    records = Medline.parse(fetch_handle)
    for record in records:
        if 'AU' in record:
            author = ','.join(record['AU'])
            print(author)
        else:
            author = ''

        if 'AID' in record:    
            doi = ','.join(record['AID'])
            print(doi)
        else:
            doi = ''

        if 'PMC' in record:    
            pmc = record['PMC']
            print(pmc)
        else:
            pmc = ''
        
        if 'PMID' in record:    
            pmcid = record['PMID']
            print(pmcid)
        else:
            pmcid = ''
        
        if 'TI' in record:    
            title = record['TI']
            print(title)
        else:
            title = ''
        
        if 'PG' in record:    
            page = record['PG']
            print(page)
        else:
            page = ''
        
        if 'AB' in record:    
            abstract = record['AB']
            print(abstract)
        else:
            abstract = ''
        
        if 'PT' in record:    
            rec_type = ','.join(record['PT'])
            print(rec_type)
        else:
            rec_type = ''
        
        if 'SO' in record:    
            so = record['SO']
            print(so)
        else:
            so = ''

        print(title + ';' + author + ';' + doi + ';' + pmc + ';' + pmcid + ';' + page + ';' + rec_type + ';' + so + ';' + abstract, file=dst)
