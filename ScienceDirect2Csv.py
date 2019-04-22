import pandas as pd

dst = open("/tmp/science_direct.csv", "w")

f = open("/home/amenegotto/Downloads/ScienceDirect_citations_1555928434451.txt", "r")
i = 1
for line in f:
    line = line.replace('\n','')
    
    print(line)

    if line == '':
        print(author + ';' + title + ';' + journal + ';' + volume + ';' + year + ';' + pages + ';' + issn + ';' + doi + ';' + link + ';' + abstract + '\n', file=dst)
        i = 1
        continue

    if i == 1:
        author = line
        print("author = " + author)
    elif i == 2:
        title = line
        print("title = " + title)
    elif i == 3:
        journal = line
        print("journal = " + journal)
    elif i == 4:
        volume = line
        print("volume = " + volume)
    elif i == 5:
        year = line
        print("year = " + year)
    elif i == 6:
        pages = line
        print("pages = " + pages)
    elif i == 7:
        issn = line
        print("issn = " + issn)
    elif i == 8:
        doi = line
        print("doi = " + doi)
    elif i == 9:
        link = line
        print("link = " + link)
    elif i == 10:
        abstract = line
        print("abstract = " + abstract)
    
    i = i + 1

dst.close()
f.close()
